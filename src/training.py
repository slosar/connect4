"""
Training pipeline for Alpha-Zero Connect4.

This module provides the main training loop that orchestrates self-play,
neural network training, and model evaluation.
"""

import os

# Setup environment before any JAX imports
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
#os.environ['JAX_PLATFORMS'] = 'cuda'  # Force GPU execution

import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import Dict, List, Tuple, Any
import time
import logging
from datetime import datetime

from neural_network import (
    Connect4Network, AlphaZeroTrainState, create_train_state, 
    train_step, compute_loss
)
from self_play import SelfPlayManager, TrainingDataBuffer, evaluate_model_strength
from alphazero_player import AlphaZeroPlayer


logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration for training."""
    
    def __init__(self):
        # Network architecture
        self.num_filters = 64
        self.num_blocks = 8
        
        # Training parameters
        self.learning_rate = 1e-3
        self.batch_size = 64
        self.training_steps_per_iteration = 1000
        self.checkpoint_frequency = 10  # Save every N iterations
        self.evaluation_frequency = 5   # Evaluate every N iterations
        
        # Self-play parameters
        self.games_per_iteration = 100
        self.num_simulations = 800
        self.num_workers = 1  # Sequential execution with GPU acceleration
        
        # Data management
        self.buffer_size = 100000
        self.min_buffer_size = 1000  # Minimum data before training
        
        # Game configurations
        self.game_configs = [
            {'rows': 5, 'cols': 5, 'connect_length': 4},
            {'rows': 6, 'cols': 6, 'connect_length': 4},
            {'rows': 6, 'cols': 7, 'connect_length': 4},  # Standard
            {'rows': 7, 'cols': 8, 'connect_length': 4},
            {'rows': 8, 'cols': 8, 'connect_length': 4},
            {'rows': 10, 'cols': 10, 'connect_length': 4},
            {'rows': 12, 'cols': 12, 'connect_length': 4},
            {'rows': 15, 'cols': 15, 'connect_length': 4},
            {'rows': 20, 'cols': 20, 'connect_length': 4},
            {'rows': 25, 'cols': 25, 'connect_length': 4},
            {'rows': 30, 'cols': 30, 'connect_length': 4},
        ]
        
        # Paths
        self.model_dir = "models"
        self.data_dir = "training_data"
        self.log_dir = "logs"


class AlphaZeroTrainer:
    """Main trainer for Alpha-Zero Connect4."""
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        
        # Initialize random key
        self.rng = jax.random.PRNGKey(42)
        
        # Create directories
        os.makedirs(self.config.model_dir, exist_ok=True)
        os.makedirs(self.config.data_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        # Initialize components
        self.train_state = None
        self.data_buffer = TrainingDataBuffer(max_size=self.config.buffer_size)
        self.self_play_manager = SelfPlayManager(
            num_workers=self.config.num_workers,
            games_per_iteration=self.config.games_per_iteration,
            num_simulations=self.config.num_simulations,
            game_configs=self.config.game_configs
        )
        
        # Training metrics
        self.iteration = 0
        self.training_history = []
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
    
    def initialize_model(self, model_path: str = None) -> None:
        """Initialize or load the neural network model."""
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading existing model from {model_path}")
            try:
                player = AlphaZeroPlayer(
                    model_path=model_path,
                    num_filters=self.config.num_filters,
                    num_blocks=self.config.num_blocks
                )
                self.train_state = player.train_state
                logger.info("Successfully loaded existing model")
            except Exception as e:
                logger.warning(f"Failed to load model: {str(e)}")
                logger.info("Creating new model instead")
                self.train_state = create_train_state(
                    self.rng,
                    learning_rate=self.config.learning_rate,
                    num_filters=self.config.num_filters,
                    num_blocks=self.config.num_blocks
                )
        else:
            logger.info("Initializing new model")
            self.train_state = create_train_state(
                self.rng,
                learning_rate=self.config.learning_rate,
                num_filters=self.config.num_filters,
                num_blocks=self.config.num_blocks
            )
    
    def run_training_iteration(self) -> Dict[str, Any]:
        """Run one training iteration (self-play + training)."""
        iteration_start = time.time()
        self.iteration += 1
        
        logger.info(f"\\n=== Training Iteration {self.iteration} ===")
        
        # 1. Self-play to generate training data
        logger.info("Running self-play...")
        current_model_path = self._get_current_model_path()
        new_data = self.self_play_manager.generate_training_data(current_model_path)
        
        # 2. Add new data to buffer
        self.data_buffer.add_data(new_data)
        logger.info(f"Training buffer size: {len(self.data_buffer)}")
        
        # 3. Train the neural network
        if len(self.data_buffer) >= self.config.min_buffer_size:
            logger.info("Training neural network...")
            training_metrics = self._train_network()
        else:
            logger.info(f"Insufficient data for training (need {self.config.min_buffer_size})")
            training_metrics = {}
        
        # 4. Evaluate model periodically
        evaluation_metrics = {}
        if self.iteration % self.config.evaluation_frequency == 0:
            logger.info("Evaluating model...")
            evaluation_metrics = self._evaluate_model()
        
        # 5. Save checkpoint periodically
        if self.iteration % self.config.checkpoint_frequency == 0:
            self._save_checkpoint()
        
        # 6. Compile iteration metrics
        iteration_time = time.time() - iteration_start
        self_play_stats = self.self_play_manager.get_statistics()
        
        iteration_metrics = {
            'iteration': self.iteration,
            'iteration_time': iteration_time,
            'buffer_size': len(self.data_buffer),
            'new_positions': len(new_data),
            'self_play_stats': self_play_stats,
            'training_metrics': training_metrics,
            'evaluation_metrics': evaluation_metrics
        }
        
        self.training_history.append(iteration_metrics)
        
        logger.info(f"Iteration {self.iteration} completed in {iteration_time:.2f}s")
        if training_metrics:
            logger.info(f"Training loss: {training_metrics.get('total_loss', 'N/A'):.4f}")
        
        return iteration_metrics
    
    def _train_network(self) -> Dict[str, float]:
        """Train the neural network on buffered data."""
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        for step in range(self.config.training_steps_per_iteration):
            # Sample batch
            board_states, policies, values, valid_masks = self.data_buffer.sample_batch(
                self.config.batch_size
            )
            
            # Convert to JAX arrays
            board_states = jnp.array(board_states)
            policies = jnp.array(policies)
            values = jnp.array(values)
            valid_masks = jnp.array(valid_masks)
            
            # Training step
            self.train_state, metrics = train_step(
                self.train_state, board_states, policies, values, valid_masks
            )
            
            # Accumulate metrics
            #print (total_loss, total_policy_loss, total_value_loss,'XX')
            total_loss += float(metrics['total_loss'])
            total_policy_loss += float(metrics['policy_loss'])
            total_value_loss += float(metrics['value_loss'])
            
            if step % 100 == 0:
                logger.info(f"  Step {step}/{self.config.training_steps_per_iteration}, "
                           f"Loss: {float(metrics['total_loss']):.4f}")
        
        # Average metrics
        num_steps = self.config.training_steps_per_iteration
        return {
            'total_loss': total_loss / num_steps,
            'policy_loss': total_policy_loss / num_steps,
            'value_loss': total_value_loss / num_steps,
            'training_steps': num_steps
        }
    
    def _evaluate_model(self) -> Dict[str, float]:
        """Evaluate current model against random player."""
        current_model_path = self._get_current_model_path()
        
        # Evaluate against random player
        results = evaluate_model_strength(
            model_path=current_model_path,
            baseline_path=None,  # Random player
            num_games=50,
            num_simulations=400
        )
        
        return results
    
    def _save_checkpoint(self) -> None:
        """Save model checkpoint and training data."""
        # Save model
        model_path = os.path.join(
            self.config.model_dir, 
            f"model_iteration_{self.iteration}.pkl"
        )
        
        # Create player with correct architecture parameters
        player = AlphaZeroPlayer(
            model_path=None,
            num_filters=self.config.num_filters,
            num_blocks=self.config.num_blocks
        )
        player.train_state = self.train_state
        player.save_model(model_path)
        
        # Save latest model (for continuing training)
        latest_path = os.path.join(self.config.model_dir, "latest_model.pkl")
        player.save_model(latest_path)
        
        # Save training data buffer
        data_path = os.path.join(
            self.config.data_dir,
            f"training_data_iteration_{self.iteration}.pkl"
        )
        self.data_buffer.save_to_disk(data_path)
        
        # Save training history
        history_path = os.path.join(
            self.config.log_dir,
            "training_history.pkl"
        )
        import pickle
        with open(history_path, 'wb') as f:
            pickle.dump(self.training_history, f)
        
        logger.info(f"Checkpoint saved: {model_path}")
    
    def _get_current_model_path(self) -> str:
        """Get path to current model for self-play."""
        latest_path = os.path.join(self.config.model_dir, "latest_model.pkl")
        if os.path.exists(latest_path):
            return latest_path
        return None
    
    def train(self, num_iterations: int = 100) -> None:
        """
        Run the complete training loop.
        
        Args:
            num_iterations: Number of training iterations to run
        """
        logger.info(f"Starting Alpha-Zero training for {num_iterations} iterations")
        logger.info(f"Configuration: {self.config.__dict__}")
        
        # Initialize model
        self.initialize_model()
        
        # IMPORTANT: Save initial model before starting self-play
        # This ensures that self-play always has a model to work with
        self._save_checkpoint()
        
        start_time = time.time()
        
        try:
            for i in range(num_iterations):
                iteration_metrics = self.run_training_iteration()
                
                # Log progress
                elapsed_time = time.time() - start_time
                avg_time_per_iteration = elapsed_time / (i + 1)
                eta = avg_time_per_iteration * (num_iterations - i - 1)
                
                logger.info(f"Progress: {i+1}/{num_iterations} iterations, "
                           f"ETA: {eta/3600:.1f} hours")
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self._save_checkpoint()
        
        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
            raise
        
        finally:
            # Final save
            self._save_checkpoint()
            
            total_time = time.time() - start_time
            logger.info(f"Training completed in {total_time/3600:.2f} hours")
    
    def resume_training(self, num_iterations: int = 100) -> None:
        """Resume training from latest checkpoint."""
        latest_model = os.path.join(self.config.model_dir, "latest_model.pkl")
        latest_data = os.path.join(self.config.data_dir, "training_data_latest.pkl")
        
        # Load model
        if os.path.exists(latest_model):
            self.initialize_model(latest_model)
            logger.info("Resumed from latest model checkpoint")
        else:
            logger.info("No checkpoint found, starting fresh")
            self.initialize_model()
        
        # Load training data
        if os.path.exists(latest_data):
            self.data_buffer.load_from_disk(latest_data)
            logger.info(f"Loaded {len(self.data_buffer)} training examples")
        
        # Load training history
        history_path = os.path.join(self.config.log_dir, "training_history.pkl")
        if os.path.exists(history_path):
            import pickle
            with open(history_path, 'rb') as f:
                self.training_history = pickle.load(f)
            self.iteration = len(self.training_history)
            logger.info(f"Resumed from iteration {self.iteration}")
        
        # Continue training
        self.train(num_iterations)


def create_training_config(
    num_filters: int = 64,
    num_blocks: int = 8,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    games_per_iteration: int = 100,
    num_simulations: int = 800,
    num_workers: int = 1  # Deprecated parameter, kept for compatibility
) -> TrainingConfig:
    """Create a training configuration with custom parameters."""
    config = TrainingConfig()
    config.num_filters = num_filters
    config.num_blocks = num_blocks
    config.learning_rate = learning_rate
    config.batch_size = batch_size
    config.games_per_iteration = games_per_iteration
    config.num_simulations = num_simulations
    config.num_workers = 1  # Always use sequential with GPU
    return config