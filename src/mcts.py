"""
Monte Carlo Tree Search implementation for Alpha-Zero style Connect4 agent.

This implementation follows the Alpha-Zero MCTS algorithm with UCB selection,
neural network evaluation, and proper backpropagation of values.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple, Any
import math
import copy
from dataclasses import dataclass

from connect4 import Connect4, GameState, Player


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    
    # State information
    game_state: Connect4
    parent: Optional['MCTSNode'] = None
    action: Optional[int] = None  # Action that led to this node
    
    # MCTS statistics
    visit_count: int = 0
    total_value: float = 0.0
    prior_probability: float = 0.0
    
    # Children nodes
    children: Dict[int, 'MCTSNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}
    
    @property
    def q_value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children expanded)."""
        return len(self.children) == 0
    
    @property
    def is_terminal(self) -> bool:
        """Check if this represents a terminal game state."""
        return self.game_state.is_game_over()
    
    def ucb_score(self, c_puct: float = 1.0) -> float:
        """Calculate UCB score for node selection."""
        if self.visit_count == 0:
            return float('inf')
        
        # UCB formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        exploitation = self.q_value
        
        parent_visits = self.parent.visit_count if self.parent else 1
        exploration = (c_puct * self.prior_probability * 
                      math.sqrt(parent_visits) / (1 + self.visit_count))
        
        return exploitation + exploration


class MCTS:
    """Monte Carlo Tree Search for Connect4."""
    
    def __init__(self, 
                 neural_network,
                 train_state,
                 c_puct: float = 1.0,
                 num_simulations: int = 800,
                 temperature: float = 1.0,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25):
        """
        Initialize MCTS.
        
        Args:
            neural_network: Neural network for evaluation
            train_state: JAX training state containing model parameters
            c_puct: UCB exploration constant
            num_simulations: Number of MCTS simulations per move
            temperature: Temperature for action selection
            dirichlet_alpha: Alpha parameter for Dirichlet noise
            dirichlet_epsilon: Weight for Dirichlet noise
        """
        self.neural_network = neural_network
        self.train_state = train_state
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
    
    def search(self, game_state: Connect4) -> Tuple[np.ndarray, List[Tuple]]:
        """
        Perform MCTS search and return action probabilities.
        
        Args:
            game_state: Current game state
            
        Returns:
            action_probs: Probability distribution over actions
            search_path: List of (state, action_probs, value) for training
        """
        # Create root node
        root = MCTSNode(game_state=copy.deepcopy(game_state))
        
        # Expand root node with neural network evaluation
        self._expand_node(root)
        
        # Add Dirichlet noise to root node for exploration
        self._add_dirichlet_noise(root)
        
        search_path = []
        
        # Perform simulations
        for _ in range(self.num_simulations):
            # Selection: traverse down the tree
            node = root
            path = [node]
            
            while not node.is_leaf and not node.is_terminal:
                action = self._select_action(node)
                node = node.children[action]
                path.append(node)
            
            # Expansion and Evaluation
            if not node.is_terminal:
                self._expand_node(node)
                
                # If node was expanded, select first child for evaluation
                if node.children:
                    action = next(iter(node.children.keys()))
                    node = node.children[action]
                    path.append(node)
            
            # Get value for leaf node
            if node.is_terminal:
                value = self._get_terminal_value(node)
            else:
                policy_logits, value = self._evaluate_with_network(node)
            
            # Backpropagation
            self._backpropagate(path, value)
        
        # Calculate action probabilities from visit counts
        action_probs = self._get_action_probabilities(root)
        
        # Store search statistics for training
        board_state = root.game_state.get_board_state()
        search_path.append((board_state, action_probs, root.q_value))
        
        return action_probs, search_path
    
    def _expand_node(self, node: MCTSNode) -> None:
        """Expand a node by creating children for all valid actions."""
        if node.is_terminal:
            return
        
        valid_actions = node.game_state.get_valid_moves()
        if not valid_actions:
            return
        
        # Get neural network predictions
        policy_logits, value = self._evaluate_with_network(node)
        
        # Apply softmax to policy logits for valid actions
        valid_policy = np.full(node.game_state.MAX_COLS, -np.inf)
        valid_policy[valid_actions] = policy_logits[valid_actions]
        policy_probs = self._softmax(valid_policy)
        
        # Create child nodes
        for action in valid_actions:
            child_game = copy.deepcopy(node.game_state)
            child_game.make_move(action)
            
            child_node = MCTSNode(
                game_state=child_game,
                parent=node,
                action=action,
                prior_probability=policy_probs[action]
            )
            
            node.children[action] = child_node
    
    def _evaluate_with_network(self, node: MCTSNode) -> Tuple[np.ndarray, float]:
        """Evaluate a node using the neural network."""
        from neural_network import apply_model
        
        board_state = node.game_state.get_board_state()
        # Ensure board state is (5, 30, 30) and add batch dimension to get (1, 5, 30, 30)
        board_state = jnp.array(board_state)
        if board_state.ndim == 3:  # (5, 30, 30)
            board_state = jnp.expand_dims(board_state, axis=0)  # Add batch dimension
        
        policy_logits, value = apply_model(self.train_state, board_state, training=False)
        
        # Convert to numpy and remove batch dimension
        policy_logits = np.array(policy_logits[0])
        value = float(value[0])
        
        # Flip value for opponent's perspective
        if node.game_state.current_player == Player.TWO:
            value = -value
        
        return policy_logits, value
    
    def _select_action(self, node: MCTSNode) -> int:
        """Select action based on UCB scores."""
        best_action = None
        best_score = -float('inf')
        
        for action, child in node.children.items():
            score = child.ucb_score(self.c_puct)
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def _get_terminal_value(self, node: MCTSNode) -> float:
        """Get value for terminal node."""
        if node.game_state.game_state == GameState.DRAW:
            return 0.0
        elif node.game_state.get_winner() == Player.ONE:
            return 1.0 if node.game_state.current_player == Player.ONE else -1.0
        else:  # Player TWO wins
            return 1.0 if node.game_state.current_player == Player.TWO else -1.0
    
    def _backpropagate(self, path: List[MCTSNode], value: float) -> None:
        """Backpropagate value up the tree."""
        for i, node in enumerate(reversed(path)):
            # Flip value for alternating players
            node_value = value * ((-1) ** i)
            node.visit_count += 1
            node.total_value += node_value
    
    def _get_action_probabilities(self, root: MCTSNode) -> np.ndarray:
        """Calculate action probabilities from visit counts."""
        action_probs = np.zeros(root.game_state.MAX_COLS)
        
        if self.temperature == 0:
            # Greedy selection
            best_action = max(root.children.keys(), 
                            key=lambda a: root.children[a].visit_count)
            action_probs[best_action] = 1.0
        else:
            # Proportional to visit counts with temperature
            visits = np.array([root.children.get(a, MCTSNode(game_state=None)).visit_count 
                              for a in range(root.game_state.MAX_COLS)])
            
            if self.temperature != 1.0:
                visits = visits ** (1.0 / self.temperature)
            
            if visits.sum() > 0:
                action_probs = visits / visits.sum()
        
        return action_probs
    
    def _add_dirichlet_noise(self, root: MCTSNode) -> None:
        """Add Dirichlet noise to root node for exploration."""
        if not root.children:
            return
        
        actions = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        
        for i, action in enumerate(actions):
            child = root.children[action]
            child.prior_probability = ((1 - self.dirichlet_epsilon) * child.prior_probability + 
                                     self.dirichlet_epsilon * noise[i])
    
    @staticmethod
    def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Compute softmax probabilities."""
        x = x / temperature
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x)


class MCTSPlayer:
    """Player that uses MCTS for move selection."""
    
    def __init__(self, mcts: MCTS, name: str = "MCTS Player"):
        self.mcts = mcts
        self.name = name
        self.search_history = []  # For training data collection
    
    def get_move(self, game_state: Connect4) -> Tuple[int, List[Tuple]]:
        """
        Get move using MCTS.
        
        Returns:
            action: Selected action
            search_data: Search data for training
        """
        action_probs, search_path = self.mcts.search(game_state)
        
        # Sample action from probabilities
        valid_actions = game_state.get_valid_moves()
        valid_probs = action_probs[valid_actions]
        valid_probs = valid_probs / valid_probs.sum()  # Renormalize
        
        action = np.random.choice(valid_actions, p=valid_probs)
        
        # Store for training
        self.search_history.extend(search_path)
        
        return action, search_path
    
    def reset_history(self):
        """Reset search history."""
        self.search_history = []
    
    def set_temperature(self, temperature: float):
        """Set temperature for action selection."""
        self.mcts.temperature = temperature