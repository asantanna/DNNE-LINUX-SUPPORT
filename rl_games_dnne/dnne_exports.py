"""
DNNE Export Interface for rl_games_dnne
This module exposes PPO components needed by DNNE's exported code
"""

import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from typing import Dict, Any, Tuple, Optional


class PPOComponents:
    """
    Standalone PPO algorithm components for DNNE exports
    Extracted from rl_games A2CBase implementation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with rl_games-compatible configuration"""
        self.e_clip = config.get('e_clip', 0.2)
        self.critic_coef = config.get('critic_coef', 4.0)
        self.entropy_coef = config.get('entropy_coef', 0.0)
        self.tau = config.get('tau', 0.95)
        self.gamma = config.get('gamma', 0.99)
        self.grad_norm = config.get('grad_norm', 1.0)
        self.learning_rate = config.get('learning_rate', 0.0003)
        self.mini_epochs_num = config.get('mini_epochs_num', 8)
        self.minibatch_size = config.get('minibatch_size', 8192)
        self.horizon_length = config.get('horizon_length', 16)
        self.clip_value = config.get('clip_value', True)
        self.bounds_loss_coef = config.get('bounds_loss_coef', 0.0001)
        self.bound_loss_type = config.get('bound_loss_type', 'bound')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ppo = True  # Always PPO mode
        
    def discount_values(self, rewards: torch.Tensor, values: torch.Tensor, 
                       dones: torch.Tensor, gamma: Optional[float] = None, 
                       tau: Optional[float] = None) -> torch.Tensor:
        """
        Compute GAE advantages using rl_games implementation
        Direct adaptation from rl_games A2CBase.discount_values()
        
        Args:
            rewards: Tensor of rewards [horizon_length, num_envs]
            values: Tensor of value predictions [horizon_length + 1, num_envs]
            dones: Tensor of done flags [horizon_length + 1, num_envs]
            gamma: Discount factor (uses self.gamma if not provided)
            tau: GAE parameter (uses self.tau if not provided)
            
        Returns:
            advantages: GAE advantages [horizon_length, num_envs]
        """
        if gamma is None:
            gamma = self.gamma
        if tau is None:
            tau = self.tau
            
        # Extract the components
        mb_rewards = rewards  # [horizon_length, num_envs]
        mb_values = values[:-1]  # [horizon_length, num_envs]
        mb_dones = dones[:-1]  # [horizon_length, num_envs]
        last_values = values[-1]  # [num_envs]
        fdones = dones[-1]  # [num_envs]
        
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)
        
        for t in reversed(range(mb_rewards.shape[0])):
            if t == mb_rewards.shape[0] - 1:
                # Convert boolean to float for arithmetic
                nextnonterminal = (~fdones).float() if fdones.dtype == torch.bool else 1.0 - fdones
                nextvalues = last_values
            else:
                # Convert boolean to float for arithmetic
                nextnonterminal = (~mb_dones[t+1]).float() if mb_dones[t+1].dtype == torch.bool else 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            
            delta = mb_rewards[t] + gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + gamma * tau * nextnonterminal * lastgaelam
            
        return mb_advs
    
    def train_actor_critic(self, input_dict: Dict[str, torch.Tensor], 
                          model: nn.Module) -> Tuple[Dict[str, float], torch.Tensor]:
        """
        Compute PPO losses for actor-critic training
        Simplified from rl_games A2CBase.train_actor_critic()
        
        Args:
            input_dict: Dictionary containing:
                - obs: Observations [batch_size, obs_dim]
                - actions: Actions taken [batch_size, action_dim]
                - old_values: Previous value predictions [batch_size]
                - advantages: GAE advantages [batch_size]
                - returns: Value targets [batch_size]
                - old_logp_actions: Previous action log probs [batch_size]
                - mu: Action means for continuous (optional) [batch_size, action_dim]
                - sigma: Action stds for continuous (optional) [batch_size, action_dim]
            model: Actor-critic model with specific structure
            
        Returns:
            info_dict: Dictionary with training metrics
            total_loss: Total loss value
        """
        # print("[DEBUG] train_actor_critic STARTED")
        obs = input_dict['obs']
        actions = input_dict['actions']
        old_values = input_dict['old_values']
        advantages = input_dict['advantages']
        returns = input_dict['returns']
        old_action_log_probs = input_dict['old_logp_actions']
        
        # For continuous actions
        old_mu = input_dict.get('mu')
        old_sigma = input_dict.get('sigma')
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass through model
        # Get shared features
        # print("[DEBUG] Getting shared features...")
        features = model['shared'](obs)
        # print(f"[DEBUG] Features shape: {features.shape}")
        
        # Get value predictions
        # print("[DEBUG] Getting value predictions...")
        values = model['value'](features).squeeze(-1)
        # print(f"[DEBUG] Values shape: {values.shape}")
        
        # Get action distribution
        # print(f"[DEBUG] Getting action distribution (old_mu is {'not ' if old_mu is None else ''}None)...")
        if old_mu is not None:  # Continuous actions
            # print("[DEBUG] Continuous action path...")
            action_mean = model['policy_mean'](features)
            # print(f"[DEBUG] Action mean shape: {action_mean.shape}")
            action_log_std = model['policy_log_std']['log_std']
            action_std = torch.exp(action_log_std)
            # print(f"[DEBUG] Action std shape: {action_std.shape}")
            
            # Create distribution
            distr = dist.Normal(action_mean, action_std)
            action_log_probs = distr.log_prob(actions).sum(dim=-1)
            entropy = distr.entropy().sum(dim=-1).mean()
            # print(f"[DEBUG] Action log probs shape: {action_log_probs.shape}, entropy: {entropy.item():.4f}")
            
        else:  # Discrete actions
            action_logits = model['policy_mean'](features)
            distr = dist.Categorical(logits=action_logits)
            action_log_probs = distr.log_prob(actions.squeeze(-1))
            entropy = distr.entropy().mean()
        
        # PPO loss
        ratio = torch.exp(action_log_probs - old_action_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        if self.clip_value:
            value_pred_clipped = old_values + torch.clamp(values - old_values, -self.e_clip, self.e_clip)
            value_losses = (values - returns)**2
            value_losses_clipped = (value_pred_clipped - returns)**2
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = 0.5 * ((values - returns)**2).mean()
        
        # Entropy bonus
        entropy_loss = -entropy
        
        # Total loss
        total_loss = actor_loss + self.critic_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Optional: Bounds loss for continuous actions
        if old_mu is not None and self.bounds_loss_coef > 0:
            if self.bound_loss_type == 'bound':
                mu = action_mean
                # Penalize actions outside [-1, 1]
                bounds_loss = torch.max(torch.abs(mu) - 1, torch.zeros_like(mu)).sum(dim=-1).mean()
            elif self.bound_loss_type == 'regularization':
                bounds_loss = torch.abs(action_mean).mean()
            else:
                bounds_loss = 0.0
            total_loss += self.bounds_loss_coef * bounds_loss
        
        # PPO_GRAD debug logging to match rl_games
        import os
        if os.environ.get('PPO_CYCLE_DEBUG', '0') == '1':
            print(f"[DNNE_DEBUG] PPO_GRAD: Batch size: {obs.shape[0]}")
            print(f"[DNNE_DEBUG] PPO_GRAD: Actor loss: {actor_loss.item():.6f}")
            print(f"[DNNE_DEBUG] PPO_GRAD: Critic loss: {value_loss.item():.6f}")
            print(f"[DNNE_DEBUG] PPO_GRAD: Entropy: {entropy.item():.6f}")
            print(f"[DNNE_DEBUG] PPO_GRAD: Total loss: {total_loss.item():.6f}")
            print(f"[DNNE_DEBUG] PPO_GRAD: Advantage mean: {advantages.mean().item():.4f}, std: {advantages.std().item():.4f}")
            print(f"[DNNE_DEBUG] PPO_GRAD: First 5 advantages: {advantages[:5].tolist()}")
            print(f"[DNNE_DEBUG] PPO_GRAD: First 5 old log probs: {old_action_log_probs[:5].tolist()}")
            print(f"[DNNE_DEBUG] PPO_GRAD: First 5 new log probs: {action_log_probs[:5].tolist()}")
            
            if old_mu is not None:  # Additional debug for continuous actions
                # Compute KL divergence
                kl_dist = torch.mean(torch.sum(
                    torch.log(old_sigma / action_std) + 
                    (action_std**2 + (old_mu - action_mean)**2) / (2.0 * old_sigma**2) - 0.5,
                    dim=-1
                ))
                print(f"[DNNE_DEBUG] PPO_GRAD: KL divergence: {kl_dist.item():.6f}")
                print(f"[DNNE_DEBUG] PPO_GRAD: Mu shape: {action_mean.shape}, mean: {action_mean.mean().item():.4f}, std: {action_mean.std().item():.4f}")
                print(f"[DNNE_DEBUG] PPO_GRAD: Sigma shape: {action_std.shape}, mean: {action_std.mean().item():.4f}, std: {action_std.std().item():.4f}")
                # Handle different tensor shapes for debug printing
                if action_mean.shape[0] > 0 and len(action_mean.shape) > 1:
                    print(f"[DNNE_DEBUG] PPO_GRAD: First 5 mu values: {action_mean[0][:5].tolist()}")
                elif action_mean.numel() > 0:
                    print(f"[DNNE_DEBUG] PPO_GRAD: Mu value: {action_mean.tolist()}")
                    
                if action_std.numel() > 5:
                    print(f"[DNNE_DEBUG] PPO_GRAD: First 5 sigma values: {action_std[:5].tolist()}")
                else:
                    print(f"[DNNE_DEBUG] PPO_GRAD: Sigma values: {action_std.tolist()}")
                    
                if old_mu.shape[0] > 0 and len(old_mu.shape) > 1:
                    print(f"[DNNE_DEBUG] PPO_GRAD: First 5 old mu values: {old_mu[0][:5].tolist()}")
                elif old_mu.numel() > 0:
                    print(f"[DNNE_DEBUG] PPO_GRAD: Old mu value: {old_mu.tolist()}")
                    
                if old_sigma.numel() > 5:
                    print(f"[DNNE_DEBUG] PPO_GRAD: First 5 old sigma values: {old_sigma[:5].tolist()}")
                else:
                    print(f"[DNNE_DEBUG] PPO_GRAD: Old sigma values: {old_sigma.tolist()}")
        
        # Info dict for logging
        info_dict = {
            'actor_loss': actor_loss.item(),
            'critic_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item(),
            'ratio': ratio.mean().item(),
            'advantages_mean': advantages.mean().item(),
            'values_mean': values.mean().item(),
        }
        
        # print(f"[DEBUG] train_actor_critic COMPLETED, total_loss={total_loss.item():.4f}")
        return info_dict, total_loss


class RunningMeanStd:
    """
    Tracks running mean and standard deviation for normalization
    Used for observation and value function normalization
    """
    
    def __init__(self, shape, epsilon=1e-4, device='cpu'):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon
        self.device = device
        
    def update(self, x):
        """Update running statistics"""
        if x.dim() > 1:
            # Flatten all dimensions except last
            x = x.view(-1, x.shape[-1])
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count
        
    def normalize(self, x):
        """Normalize input using running statistics"""
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)
    
    def denormalize(self, x):
        """Denormalize input back to original scale"""
        return x * torch.sqrt(self.var + 1e-8) + self.mean


# Export key classes/functions for easy access
__all__ = ['PPOComponents', 'RunningMeanStd']