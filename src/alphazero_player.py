"""
Alpha-Zero style player for Connect4.

This module provides a player implementation that uses MCTS with neural network
guidance for move selection, suitable for both training and gameplay.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, List, Dict, Any
import pickle
import os

from connect4 import Connect4, Player
from mcts import MCTS, MCTSPlayer
from neural_network import Connect4Network, AlphaZeroTrainState, create_train_state


class AlphaZeroPlayer:
    """
    Alpha-Zero style player for Connect4.
    
    Combines MCTS with neural network evaluation for strong gameplay
    and training data generation.
    """
    
    def __init__(self,
                 model_path: str = None,
                 num_simulations: int = 80, # was 800 AS
                 temperature: float = 1.0,
                 c_puct: float = 1.0,
                 num_filters: int = 64,
                 num_blocks: int = 8,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25):
        """
        Initialize AlphaZero player.
        
        Args:
            model_path: Path to saved model (None for random initialization)
            num_simulations: Number of MCTS simulations per move
            temperature: Temperature for action selection (1.0 for training, 0.0 for play)
            c_puct: UCB exploration constant
            num_filters: Number of filters in CNN
            num_blocks: Number of residual blocks
            dirichlet_alpha: Alpha for Dirichlet noise
            dirichlet_epsilon: Weight for Dirichlet noise
        """
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.c_puct = c_puct
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        
        # Initialize neural network
        self.rng = jax.random.PRNGKey(42)
        self.train_state = self._load_or_create_model(model_path)
        
        # Initialize MCTS
        self.mcts = MCTS(
            neural_network=Connect4Network(num_filters=num_filters, num_blocks=num_blocks),
            train_state=self.train_state,
            c_puct=c_puct,
            num_simulations=num_simulations,
            temperature=temperature,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon
        )
        
        # Create MCTS player
        self.player = MCTSPlayer(self.mcts, name="AlphaZero")
    
    def _load_or_create_model(self, model_path: str) -> AlphaZeroTrainState:
        """Load existing model or create new one."""
        if model_path and os.path.exists(model_path):
            try:
                train_state = self.load_model(model_path)
                # After loading, update MCTS with correct architecture
                self.mcts = MCTS(
                    neural_network=Connect4Network(num_filters=self.num_filters, num_blocks=self.num_blocks),
                    train_state=train_state,
                    c_puct=self.c_puct,
                    num_simulations=self.num_simulations,
                    temperature=self.temperature,
                    dirichlet_alpha=self.dirichlet_alpha,
                    dirichlet_epsilon=self.dirichlet_epsilon
                )
                self.player = MCTSPlayer(self.mcts, name="AlphaZero")
                return train_state
            except Exception as e:
                print(f"Warning: Failed to load model from {model_path}: {str(e)}")
                print("Creating new model instead...")
                return create_train_state(
                    self.rng, 
                    num_filters=self.num_filters,
                    num_blocks=self.num_blocks
                )
        else:
            return create_train_state(
                self.rng, 
                num_filters=self.num_filters,
                num_blocks=self.num_blocks
            )
    
    def get_move(self, game_state: Connect4) -> int:
        """Get move for the current game state."""
        action, _ = self.player.get_move(game_state)
        return action
    
    def get_move_with_policy(self, game_state: Connect4) -> Tuple[int, np.ndarray]:
        """Get move and return policy probabilities for training."""
        action_probs, _ = self.mcts.search(game_state)
        
        # Sample action from probabilities
        valid_actions = game_state.get_valid_moves()
        valid_probs = action_probs[valid_actions]
        valid_probs = valid_probs / valid_probs.sum()
        
        action = np.random.choice(valid_actions, p=valid_probs)
        
        return action, action_probs
    
    def evaluate_position(self, game_state: Connect4) -> Tuple[np.ndarray, float]:
        """Evaluate position with neural network (no MCTS)."""
        from neural_network import apply_model
        
        board_state = game_state.get_board_state()
        board_state = jnp.expand_dims(board_state, axis=0)
        
        policy_logits, value = apply_model(self.train_state, board_state, training=False)
        
        # Apply softmax to policy and mask invalid moves
        valid_moves = game_state.get_valid_moves()
        policy = np.full(game_state.MAX_COLS, -np.inf)
        policy[valid_moves] = np.array(policy_logits[0])[valid_moves]
        policy = self._softmax(policy)
        
        return policy, float(value[0])
    
    def set_temperature(self, temperature: float):
        """Set temperature for action selection."""
        self.temperature = temperature
        self.mcts.temperature = temperature
        self.player.set_temperature(temperature)
    
    def set_num_simulations(self, num_simulations: int):
        """Set number of MCTS simulations."""
        self.num_simulations = num_simulations
        self.mcts.num_simulations = num_simulations
    
    def save_model(self, filepath: str):
        """Save the current model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'params': self.train_state.params,
                'batch_stats': self.train_state.batch_stats,
                'num_filters': self.num_filters,
                'num_blocks': self.num_blocks
            }, f)
    
    def load_model(self, filepath: str) -> AlphaZeroTrainState:
        """Load a saved model."""
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Get architecture from saved model
        saved_filters = checkpoint.get('num_filters', 64)
        saved_blocks = checkpoint.get('num_blocks', 8)
        
        # Update instance architecture to match saved model
        self.num_filters = saved_filters
        self.num_blocks = saved_blocks
        
        # Create model with same architecture as saved model
        model = Connect4Network(
            num_filters=saved_filters,
            num_blocks=saved_blocks
        )
        
        # Create train state with loaded parameters
        # Use channels-first format: (batch, channels, height, width)
        dummy_input = jnp.ones((1, 5, 32, 32))
        variables = model.init(self.rng, dummy_input, training=True)
        
        import optax
        train_state = AlphaZeroTrainState.create(
            apply_fn=model.apply,
            params=checkpoint['params'],
            tx=optax.adam(1e-3),  # Learning rate doesn't matter for loaded model
            batch_stats=checkpoint.get('batch_stats', {})
        )
        
        return train_state
    
    def get_training_data(self) -> List[Tuple]:
        """Get training data from recent searches."""
        return self.player.search_history.copy()
    
    def reset_training_data(self):
        """Reset training data collection."""
        self.player.reset_history()
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


class RandomPlayer:
    """Random player for testing and baseline comparison."""
    
    def __init__(self, name: str = "Random"):
        self.name = name
    
    def get_move(self, game_state: Connect4) -> int:
        """Get random valid move."""
        valid_moves = game_state.get_valid_moves()
        return np.random.choice(valid_moves)
    
    def get_move_with_policy(self, game_state: Connect4) -> Tuple[int, np.ndarray]:
        """Get random move with uniform policy."""
        valid_moves = game_state.get_valid_moves()
        action = np.random.choice(valid_moves)
        
        # Create uniform policy over valid moves
        policy = np.zeros(game_state.MAX_COLS)
        policy[valid_moves] = 1.0 / len(valid_moves)
        
        return action, policy


class HumanPlayer:
    """Human player for interactive gameplay."""
    
    def __init__(self, name: str = "Human"):
        self.name = name
    
    def get_move(self, game_state: Connect4) -> int:
        """Get move from human input."""
        valid_moves = game_state.get_valid_moves()
        
        print(f"\\nCurrent board:")
        print(game_state)
        print(f"Valid moves: {valid_moves}")
        
        while True:
            try:
                move = int(input(f"Enter your move (0-{game_state.cols-1}): "))
                if move in valid_moves:
                    return move
                else:
                    print(f"Invalid move! Valid moves are: {valid_moves}")
            except ValueError:
                print("Please enter a valid integer.")
            except KeyboardInterrupt:
                print("\\nGame interrupted by user.")
                raise


def play_game(player1, player2, game_config: Dict[str, int] = None) -> Tuple[Connect4, List]:
    """
    Play a game between two players.
    
    Args:
        player1: First player
        player2: Second player  
        game_config: Game configuration (rows, cols, connect_length)
    
    Returns:
        game: Final game state
        game_history: List of (board_state, action_probs, current_player)
    """
    if game_config is None:
        game_config = {'rows': 6, 'cols': 7, 'connect_length': 4}
    
    game = Connect4(**game_config)
    players = [player1, player2]
    game_history = []
    
    while not game.is_game_over():
        current_player_obj = players[0] if game.current_player == Player.ONE else players[1]
        
        # Get move (with policy if available for training)
        if hasattr(current_player_obj, 'get_move_with_policy'):
            action, policy = current_player_obj.get_move_with_policy(game)
            # Store for training
            board_state = game.get_board_state()
            game_history.append((board_state, policy, game.current_player))
        else:
            action = current_player_obj.get_move(game)
        
        # Make move
        success = game.make_move(action)
        if not success:
            raise ValueError(f"Invalid move {action} by {current_player_obj.name}")
        print(str(game))
    
    return game, game_history