"""
Self-play system for Alpha-Zero Connect4 training.

This module provides functionality to run Connect4 games sequentially
with GPU-accelerated neural network evaluation.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Tuple, Dict, Any
import time
import os
import pickle
import logging
import traceback
import random

from connect4 import Connect4, Player, GameState


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def play_self_play_game(model_path: str, game_config: Dict, num_simulations: int, game_id: int) -> Tuple[List, int, float]:
    """
    Play a single self-play game.
    
    Args:
        model_path: Path to the model
        game_config: Game configuration dictionary
        num_simulations: Number of MCTS simulations per move
        game_id: Game identifier for logging
    
    Returns:
        training_data: List of (board_state, policy, value) tuples
        winner: Winner of the game (1, -1, or 0 for draw)
        game_length: Number of moves in the game
    """
    
    try:
        # Import here to ensure clean initialization
        from alphazero_player import AlphaZeroPlayer, play_game
        
        # Create two identical players for self-play
        player1 = AlphaZeroPlayer(
            model_path=model_path,
            num_simulations=num_simulations,
            temperature=1.0,  # Exploration during training
            c_puct=1.0
        )
        
        player2 = AlphaZeroPlayer(
            model_path=model_path,
            num_simulations=num_simulations,
            temperature=1.0,
            c_puct=1.0
        )
        
        # Play the game
        game, game_history = play_game(player1, player2, game_config)
        
        # Determine winner
        if game.game_state == GameState.DRAW:
            winner = 0
        elif game.get_winner() == Player.ONE:
            winner = 1
        else:
            winner = -1
        
        # Process training data
        training_data = []
        game_length = len(game_history)
        
        # Assign values based on game outcome
        for i, (board_state, policy, current_player) in enumerate(game_history):
            # Value from perspective of current player
            if winner == 0:
                value = 0.0  # Draw
            elif ((winner == 1 and current_player == Player.ONE) or 
                  (winner == -1 and current_player == Player.TWO)):
                value = 1.0  # Win
            else:
                value = -1.0  # Loss
            
            training_data.append((board_state, policy, value))
        
        logger.info(f"Game {game_id} completed: Winner={winner}, Length={game_length}")
        
        return training_data, winner, game_length
    
    except Exception as e:
        logger.error(f"Error in game {game_id}: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return [], 0, 0


class SelfPlayManager:
    """Manager for self-play games with GPU-accelerated neural network evaluation."""
    
    def __init__(self,
                 num_workers: int = 1,  # Not used anymore, kept for compatibility
                 games_per_iteration: int = 100,
                 num_simulations: int = 800,
                 game_configs: List[Dict] = None):
        """
        Initialize self-play manager.
        
        Args:
            num_workers: Deprecated, kept for backward compatibility
            games_per_iteration: Number of games per training iteration
            num_simulations: Number of MCTS simulations per move
            game_configs: List of game configurations to sample from
        """
        self.num_workers = 1  # Sequential execution with GPU
        self.games_per_iteration = games_per_iteration
        self.num_simulations = num_simulations
        
        # Default game configurations (various board sizes)
        if game_configs is None:
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
        else:
            self.game_configs = game_configs
        
        self.stats = {
            'total_games': 0,
            'wins_player1': 0,
            'wins_player2': 0,
            'draws': 0,
            'avg_game_length': 0.0,
            'total_positions': 0
        }
    
    def generate_training_data(self, model_path: str = None) -> List[Tuple]:
        """
        Generate training data through sequential self-play with GPU acceleration.
        
        Args:
            model_path: Path to current model
            
        Returns:
            training_data: List of (board_state, policy, value) tuples
        """
        logger.info(f"Starting {self.games_per_iteration} self-play games (JAX BACKEND={jax.devices()})")
        
        # Run games sequentially (GPU will handle batching internally)
        all_training_data = []
        start_time = time.time()
        
        for i in range(self.games_per_iteration):
            game_config = random.choice(self.game_configs)
            
            # Play game
            training_data, winner, game_length = play_self_play_game(
                model_path, game_config, self.num_simulations, i
            )
            
            # Add to dataset
            all_training_data.extend(training_data)
            
            # Update statistics
            self.stats['total_games'] += 1
            if winner == 1:
                self.stats['wins_player1'] += 1
            elif winner == -1:
                self.stats['wins_player2'] += 1
            else:
                self.stats['draws'] += 1
            
            self.stats['total_positions'] += len(training_data)
            if self.stats['total_games'] > 0:
                self.stats['avg_game_length'] = (
                    (self.stats['avg_game_length'] * (self.stats['total_games'] - 1) + game_length) / 
                    self.stats['total_games']
                )
            
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{self.games_per_iteration} games")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Generated {len(all_training_data)} training positions from "
                   f"{self.games_per_iteration} games in {elapsed_time:.2f} seconds")
        
        return all_training_data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        if self.stats['total_games'] > 0:
            stats = self.stats.copy()
            stats['win_rate_player1'] = stats['wins_player1'] / stats['total_games']
            stats['win_rate_player2'] = stats['wins_player2'] / stats['total_games']
            stats['draw_rate'] = stats['draws'] / stats['total_games']
            return stats
        return self.stats
    
    def reset_statistics(self):
        """Reset statistics."""
        self.stats = {
            'total_games': 0,
            'wins_player1': 0,
            'wins_player2': 0,
            'draws': 0,
            'avg_game_length': 0.0,
            'total_positions': 0
        }


class TrainingDataBuffer:
    """Buffer to store and manage training data."""
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.data = []
    
    def add_data(self, new_data: List[Tuple]):
        """Add new training data to buffer."""
        self.data.extend(new_data)
        
        # Keep only the most recent data
        if len(self.data) > self.max_size:
            self.data = self.data[-self.max_size:]
    
    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of training data.
        
        Returns:
            board_states: (batch_size, 5, 32, 32)
            policies: (batch_size, 32)
            values: (batch_size,)
            valid_moves_mask: (batch_size, 32)
        """
        if len(self.data) < batch_size:
            batch_size = len(self.data)
        
        indices = np.random.choice(len(self.data), batch_size, replace=False)
        
        board_states = []
        policies = []
        values = []
        valid_moves_masks = []
        
        for idx in indices:
            board_state, policy, value = self.data[idx]
            
            board_states.append(board_state)
            policies.append(policy)
            values.append(value)
            
            # Create valid moves mask from board state (layer 4)
            valid_mask = np.any(board_state[4] > 0, axis=0)  # Valid if any row in column is valid
            valid_moves_masks.append(valid_mask)
        
        return (np.array(board_states), 
                np.array(policies), 
                np.array(values), 
                np.array(valid_moves_masks))
    
    def save_to_disk(self, filepath: str):
        """Save training data to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.data, f)
    
    def load_from_disk(self, filepath: str):
        """Load training data from disk."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)


def evaluate_model_strength(model_path: str, 
                          baseline_path: str = None,
                          num_games: int = 50,
                          num_simulations: int = 400) -> Dict[str, float]:
    """
    Evaluate model strength against a baseline.
    
    Args:
        model_path: Path to model to evaluate
        baseline_path: Path to baseline model (None for random)
        num_games: Number of evaluation games
        num_simulations: Number of MCTS simulations
    
    Returns:
        results: Dictionary with win rates and other metrics
    """
    logger.info(f"Evaluating model strength over {num_games} games")
    
    # Import here to avoid JAX initialization issues
    from alphazero_player import AlphaZeroPlayer, RandomPlayer, play_game
    
    # Create players
    player1 = AlphaZeroPlayer(
        model_path=model_path,
        num_simulations=num_simulations,
        temperature=0.0  # No exploration during evaluation
    )
    
    if baseline_path:
        player2 = AlphaZeroPlayer(
            model_path=baseline_path,
            num_simulations=num_simulations,
            temperature=0.0
        )
    else:
        player2 = RandomPlayer()
    
    # Play games
    wins = 0
    draws = 0
    losses = 0
    
    game_config = {'rows': 6, 'cols': 7, 'connect_length': 4}
    
    for i in range(num_games):
        # Alternate who goes first
        if i % 2 == 0:
            game, _ = play_game(player1, player2, game_config)
            if game.get_winner() == Player.ONE:
                wins += 1
            elif game.game_state == GameState.DRAW:
                draws += 1
            else:
                losses += 1
        else:
            game, _ = play_game(player2, player1, game_config)
            if game.get_winner() == Player.TWO:
                wins += 1
            elif game.game_state == GameState.DRAW:
                draws += 1
            else:
                losses += 1
    
    win_rate = wins / num_games
    draw_rate = draws / num_games
    loss_rate = losses / num_games
    
    logger.info(f"Evaluation complete: Win rate = {win_rate:.3f}, "
               f"Draw rate = {draw_rate:.3f}, Loss rate = {loss_rate:.3f}")
    
    return {
        'win_rate': win_rate,
        'draw_rate': draw_rate,
        'loss_rate': loss_rate,
        'wins': wins,
        'draws': draws,
        'losses': losses,
        'total_games': num_games
    }