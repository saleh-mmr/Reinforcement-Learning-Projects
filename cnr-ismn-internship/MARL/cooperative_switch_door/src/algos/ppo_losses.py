# src/algos/ppo_losses.py
import torch
from torch.distributions import Categorical


def ppo_actor_loss(
    dist: Categorical,
    actions: torch.Tensor,         # [B]
    old_log_probs: torch.Tensor,   # [B]
    advantages: torch.Tensor,      # [B]
    clip_eps: float = 0.2,
    ent_coef: float = 0.01,
):
    """
    PPO clipped policy loss with entropy bonus.
    Returns: (total_loss, policy_loss, entropy)
    """
    new_log_probs = dist.log_prob(actions)               # [B]
    ratio = torch.exp(new_log_probs - old_log_probs)     # [B]

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.mean(torch.min(surr1, surr2))

    entropy = torch.mean(dist.entropy())
    total_loss = policy_loss - ent_coef * entropy
    return total_loss, policy_loss, entropy


def critic_value_loss(
    values_pred: torch.Tensor,     # [B] or [B,1]
    returns: torch.Tensor,         # [B]
):
    """
    MSE value loss for critic.
    Returns: value_loss
    """
    if values_pred.dim() == 2 and values_pred.size(-1) == 1:
        values_pred = values_pred.squeeze(-1)
    return torch.mean((values_pred - returns) ** 2)
