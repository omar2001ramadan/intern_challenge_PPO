"""PPO update code for structured policy-conditioned placement actions."""

from dataclasses import dataclass
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass
class Transition:
    graph: dict
    action: object
    old_group_logprobs: dict
    group_token_counts: dict
    value: torch.Tensor
    reward: float
    done: bool
    temperature: float


def detach_graph(graph):
    return {key: value.detach().clone() if torch.is_tensor(value) else value for key, value in graph.items()}


def detach_action(action):
    """Clone tensor fields from a PlacementPolicyAction for PPO storage."""
    kwargs = {}
    for key, value in action.__dict__.items():
        if torch.is_tensor(value):
            kwargs[key] = value.detach().clone()
        elif isinstance(value, dict):
            kwargs[key] = {
                item_key: item_value.detach().clone() if torch.is_tensor(item_value) else item_value
                for item_key, item_value in value.items()
            }
        else:
            kwargs[key] = value
    return type(action)(**kwargs)


def compute_gae(transitions, gamma=0.99, gae_lambda=0.95):
    advantages = torch.zeros(len(transitions), dtype=torch.float32)
    returns = torch.zeros(len(transitions), dtype=torch.float32)
    next_value = 0.0
    next_advantage = 0.0

    for idx in reversed(range(len(transitions))):
        transition = transitions[idx]
        value = float(transition.value.detach().item())
        mask = 0.0 if transition.done else 1.0
        delta = transition.reward + gamma * next_value * mask - value
        advantage = delta + gamma * gae_lambda * mask * next_advantage
        advantages[idx] = advantage
        returns[idx] = advantage + value
        next_value = value
        next_advantage = advantage
        if transition.done:
            next_value = 0.0
            next_advantage = 0.0

    advantages = (advantages - advantages.mean()) / torch.clamp(advantages.std(unbiased=False), min=1e-6)
    return advantages, returns


def synchronize_gradients(module):
    if not (dist.is_available() and dist.is_initialized()):
        return
    world_size = dist.get_world_size()
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def ppo_update(
    policy,
    optimizer,
    transitions,
    *,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    kl_coef=0.01,
    equivariance_coef=0.001,
    update_epochs=4,
    minibatch_size=8,
    sync_gradients=False,
):
    if not transitions:
        return {}

    advantages, returns = compute_gae(transitions, gamma=gamma, gae_lambda=gae_lambda)
    device = next(policy.parameters()).device
    gae_variance = float(advantages.var(unbiased=False).item()) if advantages.numel() > 0 else 0.0
    return_variance = float(returns.var(unbiased=False).item()) if returns.numel() > 0 else 0.0
    advantages = advantages.to(device)
    returns = returns.to(device)

    metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clipfrac": 0.0,
        "aggregate_log_ratio": 0.0,
        "aggregate_ratio": 0.0,
        "aggregate_kl": 0.0,
        "aggregate_clipfrac": 0.0,
        "equivariance_loss": 0.0,
        "gae_variance": gae_variance,
        "return_variance": return_variance,
        "updates": 0,
    }
    group_metrics = {}

    indices = list(range(len(transitions)))
    for _ in range(update_epochs):
        random.shuffle(indices)
        for start in range(0, len(indices), minibatch_size):
            batch_indices = indices[start : start + minibatch_size]
            losses = []
            policy_losses = []
            value_losses = []
            entropies = []
            kls = []
            clipfracs = []
            equivariance_losses = []

            for idx in batch_indices:
                transition = transitions[idx]
                group_logprobs, group_entropies, value, group_token_counts = policy.evaluate_action(
                    transition.graph,
                    transition.action,
                    temperature=transition.temperature,
                )
                advantage = advantages[idx]
                policy_loss_terms = []
                entropy_terms = []
                kl_terms = []
                clipfrac_terms = []
                aggregate_log_ratio = torch.zeros((), dtype=advantage.dtype, device=device)
                aggregate_old_new_kl = torch.zeros((), dtype=advantage.dtype, device=device)
                for group, logprob in group_logprobs.items():
                    old_logprob = transition.old_group_logprobs[group].detach()
                    aggregate_log_ratio = aggregate_log_ratio + (logprob - old_logprob)
                    aggregate_old_new_kl = aggregate_old_new_kl + (old_logprob - logprob)
                    token_count = float(transition.group_token_counts.get(group, group_token_counts[group]).detach().item())
                    normalizer = max(token_count, 1.0) ** 0.5
                    log_ratio = (logprob - old_logprob) / normalizer
                    ratio = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))
                    unclipped = ratio * advantage
                    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
                    group_policy_loss = -torch.minimum(unclipped, clipped)
                    group_entropy = group_entropies[group] / normalizer
                    group_kl = (old_logprob - logprob) / normalizer
                    group_clipfrac = (torch.abs(ratio.detach() - 1.0) > clip_epsilon).float()

                    policy_loss_terms.append(group_policy_loss)
                    entropy_terms.append(group_entropy)
                    kl_terms.append(group_kl)
                    clipfrac_terms.append(group_clipfrac)

                    group_metrics.setdefault(group, {"kl": 0.0, "entropy": 0.0, "clipfrac": 0.0, "count": 0})
                    group_metrics[group]["kl"] += group_kl.detach().item()
                    group_metrics[group]["entropy"] += group_entropy.detach().item()
                    group_metrics[group]["clipfrac"] += group_clipfrac.detach().item()
                    group_metrics[group]["count"] += 1

                aggregate_ratio = torch.exp(torch.clamp(aggregate_log_ratio, min=-20.0, max=20.0))
                aggregate_clipfrac = (torch.abs(aggregate_ratio.detach() - 1.0) > clip_epsilon).float()
                policy_loss = torch.stack(policy_loss_terms).mean()
                entropy = torch.stack(entropy_terms).sum()
                approx_kl = torch.stack(kl_terms).mean()
                clipfrac = torch.stack(clipfrac_terms).mean()
                value_loss = F.mse_loss(value, returns[idx])
                if equivariance_coef > 0.0 and hasattr(policy, "equivariance_loss"):
                    equivariance_loss = policy.equivariance_loss(transition.graph)
                else:
                    equivariance_loss = value_loss * 0.0
                loss = (
                    policy_loss
                    + value_coef * value_loss
                    - entropy_coef * entropy
                    + kl_coef * approx_kl
                    + equivariance_coef * equivariance_loss
                )

                losses.append(loss)
                policy_losses.append(policy_loss.detach())
                value_losses.append(value_loss.detach())
                entropies.append(entropy.detach())
                kls.append(approx_kl.detach())
                clipfracs.append(clipfrac.detach())
                equivariance_losses.append(equivariance_loss.detach())
                metrics["aggregate_log_ratio"] += aggregate_log_ratio.detach().item()
                metrics["aggregate_ratio"] += aggregate_ratio.detach().item()
                metrics["aggregate_kl"] += aggregate_old_new_kl.detach().item()
                metrics["aggregate_clipfrac"] += aggregate_clipfrac.detach().item()

            optimizer.zero_grad(set_to_none=True)
            total_loss = torch.stack(losses).mean()
            total_loss.backward()
            if sync_gradients:
                synchronize_gradients(policy)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            metrics["policy_loss"] += torch.stack(policy_losses).mean().item()
            metrics["value_loss"] += torch.stack(value_losses).mean().item()
            metrics["entropy"] += torch.stack(entropies).mean().item()
            metrics["approx_kl"] += torch.stack(kls).mean().item()
            metrics["clipfrac"] += torch.stack(clipfracs).mean().item()
            metrics["equivariance_loss"] += torch.stack(equivariance_losses).mean().item()
            metrics["updates"] += 1

    denom = max(metrics["updates"], 1)
    transition_updates = max(sum(values["count"] for values in group_metrics.values()) / max(len(group_metrics), 1), 1)
    result = {}
    for key, value in metrics.items():
        if key in {"updates", "gae_variance", "return_variance"}:
            result[key] = value
        elif key.startswith("aggregate_"):
            result[key] = value / transition_updates
        else:
            result[key] = value / denom
    for group, values in group_metrics.items():
        count = max(values["count"], 1)
        result[f"kl_{group}"] = values["kl"] / count
        result[f"entropy_{group}"] = values["entropy"] / count
        result[f"clipfrac_{group}"] = values["clipfrac"] / count
    return result
