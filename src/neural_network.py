"""
Alpha-Zero inspired neural network for Connect4.

A ResNet-style CNN with policy and value heads designed to work with
arbitrary board sizes (5x5 to 30x30) through a fixed 30x30 representation.
"""

from typing import Any, Dict, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state


class ResidualBlock(nn.Module):
    """Residual block with Conv -> BN -> ReLU -> Conv -> BN -> skip -> ReLU."""
    
    features: int = 16
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        residual = x
        
        # First conv layer
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding=1, name='conv1')(x)
        x = nn.BatchNorm(use_running_average=not training, name='bn1')(x)
        x = nn.relu(x)
        
        # Second conv layer
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding=1, name='conv2')(x)
        x = nn.BatchNorm(use_running_average=not training, name='bn2')(x)
        
        # Skip connection
        x = x + residual
        x = nn.relu(x)
        
        return x


class Connect4Network(nn.Module):
    """
    Alpha-Zero style network for Connect4.
    
    Architecture:
    - Initial conv layer to transform input channels to feature maps
    - Stack of residual blocks
    - Policy head: outputs probabilities for each column (32 outputs)
    - Value head: outputs value estimate [-1, 1]
    """
    
    num_filters: int = 16
    num_blocks: int = 8
    max_cols: int = 32
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # Initial convolution to transform input channels (5) to feature maps
        # Input shape: (batch_size, 5, 32, 32) - channels first format
        x = nn.Conv(features=self.num_filters, kernel_size=(3, 3), padding=1, name='initial_conv')(x)
        x = nn.BatchNorm(use_running_average=not training, name='initial_bn')(x)
        x = nn.relu(x)
        
        # Stack of residual blocks with explicit names
        for i in range(self.num_blocks):
            x = ResidualBlock(features=self.num_filters, name=f'residual_block_{i}')(x, training=training)
        
        # Policy head
        policy = nn.Conv(features=2, kernel_size=(1, 1), name='policy_conv')(x)
        policy = nn.BatchNorm(use_running_average=not training, name='policy_bn')(policy)
        policy = nn.relu(policy)
        policy = jnp.mean(policy, axis=(2, 3))  # Global average pooling over spatial dims
        policy = nn.Dense(features=self.max_cols, name='policy_output')(policy)  # Output for each column
        
        # Value head
        value = nn.Conv(features=1, kernel_size=(1, 1), name='value_conv')(x)
        value = nn.BatchNorm(use_running_average=not training, name='value_bn')(value)
        value = nn.relu(value)
        value = jnp.mean(value, axis=(2, 3))  # Global average pooling
        value = nn.Dense(features=64, name='value_hidden')(value)
        value = nn.relu(value)
        value = nn.Dense(features=1, name='value_output')(value)
        value = nn.tanh(value)  # Output in [-1, 1]
        value = jnp.squeeze(value, axis=-1)  # Remove last dimension
        
        return policy, value


class AlphaZeroTrainState(train_state.TrainState):
    """Training state with additional metrics tracking."""
    
    batch_stats: Any = None
    
    def apply_gradients(self, *, grads, batch_stats=None, **kwargs):
        """Apply gradients and update batch stats."""
        new_state = super().apply_gradients(grads=grads, **kwargs)
        if batch_stats is not None:
            new_state = new_state.replace(batch_stats=batch_stats)
        return new_state


def create_train_state(rng: jax.random.PRNGKey, 
                      learning_rate: float = 1e-3,
                      num_filters: int = 64,
                      num_blocks: int = 8) -> AlphaZeroTrainState:
    """Create initial training state."""
    
    model = Connect4Network(num_filters=num_filters, num_blocks=num_blocks)
    
    # Initialize with dummy input (batch_size=1, channels=5, height=32, width=32)
    # JAX uses channels-first format: (batch, channels, height, width)
    dummy_input = jnp.ones((1, 5, 32, 32))
    
    variables = model.init(rng, dummy_input, training=True)
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})
    
    optimizer = optax.adam(learning_rate)
    
    return AlphaZeroTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        batch_stats=batch_stats
    )


#@jax.jit


def apply_model(state: AlphaZeroTrainState, 
                board_state: jnp.ndarray, 
                training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply the model to get policy and value predictions."""
    
    if state.batch_stats:
        # When batch_stats exist, we need to pass them even for inference
        # Don't use mutable parameter when we don't want mutations
        policy, value = state.apply_fn(
            {'params': state.params, 'batch_stats': state.batch_stats},
            board_state,
            training=training
        )
    else:
        policy, value = state.apply_fn(
            {'params': state.params},
            board_state,
            training=training
        )
    
    return policy, value


@jax.jit
def compute_loss_jit(params, 
                     batch_stats,
                     apply_fn,
                     board_states: jnp.ndarray,
                     target_policies: jnp.ndarray,
                     target_values: jnp.ndarray,
                     valid_moves_mask: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """JIT-compiled loss computation."""
    
    if batch_stats:
        (pred_policies, pred_values), new_batch_stats = apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            board_states,
            training=True,
            mutable=['batch_stats']
        )
    else:
        pred_policies, pred_values = apply_fn(
            {'params': params},
            board_states,
            training=True
        )
        new_batch_stats = {}
    
    # Apply softmax to policy predictions and mask invalid moves
    pred_policies = jnp.where(valid_moves_mask, pred_policies, -jnp.inf)
    pred_policies = nn.softmax(pred_policies, axis=-1)
    
    # Policy loss (cross-entropy)
    policy_loss = -jnp.sum(target_policies * jnp.log(pred_policies + 1e-8), axis=-1)
    policy_loss = jnp.mean(policy_loss)
    
    # Value loss (MSE)
    value_loss = jnp.mean((pred_values - target_values) ** 2)
    
    # Total loss
    total_loss = policy_loss + value_loss
    
    metrics = {
        'total_loss': total_loss,
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'value_accuracy': jnp.mean(jnp.abs(pred_values - target_values))
    }
    
    return total_loss, (metrics, new_batch_stats)


def compute_loss(state: AlphaZeroTrainState,
                board_states: jnp.ndarray,
                target_policies: jnp.ndarray,
                target_values: jnp.ndarray,
                valid_moves_mask: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute Alpha-Zero loss function.
    
    Args:
        state: Training state
        board_states: Batch of board states (batch_size, 5, 32, 32)
        target_policies: Target policy distributions (batch_size, 32)
        target_values: Target values (batch_size,)
        valid_moves_mask: Mask for valid moves (batch_size, 32)
    
    Returns:
        loss: Total loss
        metrics: Dictionary of loss components
    """
    
    grad_fn = jax.value_and_grad(compute_loss_jit, has_aux=True)
    (loss, (metrics, new_batch_stats)), grads = grad_fn(
        state.params, 
        state.batch_stats,
        state.apply_fn,
        board_states, 
        target_policies, 
        target_values, 
        valid_moves_mask
    )
    
    return loss, grads, metrics, new_batch_stats


@jax.jit
def train_step(state: AlphaZeroTrainState,
               board_states: jnp.ndarray,
               target_policies: jnp.ndarray,
               target_values: jnp.ndarray,
               valid_moves_mask: jnp.ndarray) -> Tuple[AlphaZeroTrainState, Dict[str, jnp.ndarray]]:
    """Perform one training step."""
    
    loss, grads, metrics, new_batch_stats = compute_loss(
        state, board_states, target_policies, target_values, valid_moves_mask
    )
    
    new_state = state.apply_gradients(grads=grads, batch_stats=new_batch_stats)
    
    return new_state, metrics