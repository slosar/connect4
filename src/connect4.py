"""
Connect-4 Game Implementation

A flexible Connect-4 implementation that supports arbitrary board sizes.
Players take turns dropping pieces into columns, with the goal of connecting
4 pieces in a row (horizontally, vertically, or diagonally).
"""

from typing import List, Tuple, Optional, Literal
from enum import Enum
import numpy as np

# Try to import Numba for JIT optimization, fall back to regular Python if not available
try:
    from numba import jit, types
    from numba.typed import List as NumbaList
    NUMBA_AVAILABLE = True
except ImportError:
    # Create a dummy decorator that does nothing if Numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    NUMBA_AVAILABLE = False

class Player(Enum):
    """Enumeration for players in the game."""
    ONE = 1
    TWO = 2


class GameState(Enum):
    """Enumeration for game states."""
    IN_PROGRESS = "in_progress"
    PLAYER_ONE_WINS = "player_one_wins"
    PLAYER_TWO_WINS = "player_two_wins"
    DRAW = "draw"


# JIT-compiled utility functions for performance optimization
@jit(nopython=True, cache=True)
def _jit_check_win(board: np.ndarray, row: int, col: int, player_value: int, 
                   connect_length: int, rows: int, cols: int) -> bool:
    """
    JIT-compiled function to check if placing a piece results in a win.
    
    Args:
        board: The game board as numpy array
        row: Row of the placed piece
        col: Column of the placed piece
        player_value: Value representing the current player (1 or 2)
        connect_length: Number of pieces needed to win
        rows: Number of rows in the board
        cols: Number of columns in the board
        
    Returns:
        bool: True if the current player has won, False otherwise
    """
    # Check all four directions: horizontal, vertical, diagonal
    directions = np.array([(0, 1), (1, 0), (1, 1), (1, -1)])
    
    for i in range(4):
        dr, dc = directions[i]
        count = 1  # Count the piece just placed
        
        # Check in positive direction
        r, c = row + dr, col + dc
        while (0 <= r < rows and 0 <= c < cols and board[r, c] == player_value):
            count += 1
            r, c = r + dr, c + dc
        
        # Check in negative direction
        r, c = row - dr, col - dc
        while (0 <= r < rows and 0 <= c < cols and board[r, c] == player_value):
            count += 1
            r, c = r - dr, c - dc
        
        if count >= connect_length:
            return True
    
    return False


@jit(nopython=True, cache=True)
def _jit_get_valid_moves(board: np.ndarray, cols: int) -> np.ndarray:
    """
    JIT-compiled function to get all valid column indices where a piece can be dropped.
    
    Args:
        board: The game board as numpy array
        cols: Number of columns in the board
        
    Returns:
        np.ndarray: Array of valid column indices
    """
    valid_moves = []
    for col in range(cols):
        if board[0, col] == 0:  # Top row is empty
            valid_moves.append(col)
    
    return np.array(valid_moves)


@jit(nopython=True, cache=True)
def _jit_is_valid_move(board: np.ndarray, col: int, cols: int) -> bool:
    """
    JIT-compiled function to check if a move to the specified column is valid.
    
    Args:
        board: The game board as numpy array
        col: Column index (0-based)
        cols: Number of columns in the board
        
    Returns:
        bool: True if the move is valid, False otherwise
    """
    if col < 0 or col >= cols:
        return False
    return board[0, col] == 0


@jit(nopython=True, cache=True)
def _jit_make_move(board: np.ndarray, col: int, player_value: int, rows: int) -> int:
    """
    JIT-compiled function to make a move by dropping a piece in the specified column.
    
    Args:
        board: The game board as numpy array
        col: Column index (0-based)
        player_value: Value representing the current player (1 or 2)
        rows: Number of rows in the board
        
    Returns:
        int: Row where the piece was placed, or -1 if move is invalid
    """
    # Find the lowest empty row in the column
    for row in range(rows - 1, -1, -1):
        if board[row, col] == 0:
            board[row, col] = player_value
            return row
    return -1


@jit(nopython=True, cache=True)
def _jit_update_nn_state(board: np.ndarray, nn_state: np.ndarray, rows: int, cols: int,
                        row_ofs: int, col_ofs: int, current_player_value: int) -> None:
    """
    JIT-compiled function to update the neural network state representation.
    
    Args:
        board: The game board as numpy array
        nn_state: The neural network state array
        rows: Number of rows in the board
        cols: Number of columns in the board
        row_ofs: Row offset for centering
        col_ofs: Column offset for centering
        current_player_value: Current player value (1 or 2)
    """
    # Update player 1 and player 2 layers
    for r in range(rows):
        for c in range(cols):
            nn_state[0, row_ofs + r, col_ofs + c] = 1 if board[r, c] == 1 else 0
            nn_state[1, row_ofs + r, col_ofs + c] = 1 if board[r, c] == 2 else 0
    
    # Find the lowest empty row in each column for layer 2
    nn_state[2, :, :] = 0
    for c in range(cols):
        r = rows - 1
        while r >= 0 and board[r, c] != 0:
            r -= 1
        if r >= 0:
            nn_state[2, row_ofs + r, col_ofs + c] = 1
    
    # Set current player layer
    nn_state[3, :, :] = 1 if current_player_value == 1 else -1


class Connect4:
    """
    Connect-4 game implementation for arbitrary board sizes.
    
    The board is represented as a 2D list where:
    - 0 represents an empty cell
    - 1 represents player 1's piece
    - 2 represents player 2's piece
    
    Attributes:
        rows (int): Number of rows in the board
        cols (int): Number of columns in the board
        connect_length (int): Number of pieces needed to win (default: 4)
        board (List[List[int]]): The game board
        current_player (Player): The current player to move
        game_state (GameState): Current state of the game
    """
    
    MAX_COLS = 32
    MAX_ROWS = 32

    
    def __init__(self, rows: int = 6, cols: int = 7, connect_length: int = 4):
        """
        Initialize a new Connect-4 game.
        
        Args:
            rows (int): Number of rows in the board (default: 6)
            cols (int): Number of columns in the board (default: 7)
            connect_length (int): Number of pieces needed to win (default: 4)
            
        Raises:
            ValueError: If rows, cols, or connect_length are less than the minimum required
        """
        if rows < 1 or cols < 1:
            raise ValueError("Board dimensions must be at least 1x1")
        if connect_length < 1:
            raise ValueError("Connect length must be at least 1")
        if connect_length > max(rows, cols):
            raise ValueError("Connect length cannot exceed board dimensions")
        if rows > self.MAX_ROWS or cols > self.MAX_COLS:
            raise ValueError(f"Board dimensions cannot exceed {self.MAX_ROWS}x{self.MAX_COLS}")
        
        self.rows = rows
        self.cols = cols
        self.connect_length = connect_length
        self.board = np.zeros((rows, cols), dtype=int)
        self.current_player = Player.ONE
        self.game_state = GameState.IN_PROGRESS
        self._nn_state = np.zeros((5, self.MAX_ROWS, self.MAX_COLS), dtype=int)
        self.row_ofs = (self.MAX_ROWS - self.rows) // 2
        self.col_ofs = (self.MAX_COLS - self.cols) // 2
        self.row_ofs_end = self.row_ofs + self.rows
        self.col_ofs_end = self.col_ofs + self.cols        
        self._nn_state[4, :, :] = 0  # Valid move layer
        self._nn_state[4, self.row_ofs: self.row_ofs_end, self.col_ofs:self.col_ofs_end] = 1  # Bottom row valid
        
        # Store JIT availability for performance information
        self._jit_enabled = NUMBA_AVAILABLE
        print ("NUMBA JIT Enabled:", self._jit_enabled)

    def get_board_state(self) -> List:
        """
        Get the current state of the board for training.
        
        Returns:
            numpy array of shape (5, MAX_ROWS, MAX_COLS)
        """
        _jit_update_nn_state(self.board, self._nn_state, self.rows, self.cols,
                            self.row_ofs, self.col_ofs, self.current_player.value)
        return self._nn_state

    def get_board(self) -> List[List[int]]:
        """
        Get a copy of the current board state.
        
        Returns:
            List[List[int]]: A copy of the board
        """
        return self.board.tolist()
    
    def get_valid_moves(self) -> List[int]:
        """
        Get all valid column indices where a piece can be dropped.
        
        Returns:
            List[int]: List of valid column indices (0-based)
        """
        if self.game_state != GameState.IN_PROGRESS:
            return []
        
        valid_moves_array = _jit_get_valid_moves(self.board, self.cols)
        return valid_moves_array.tolist()
    
    def is_valid_move(self, col: int) -> bool:
        """
        Check if a move to the specified column is valid.
        
        Args:
            col (int): Column index (0-based)
            
        Returns:
            bool: True if the move is valid, False otherwise
        """
        if self.game_state != GameState.IN_PROGRESS:
            return False
        return _jit_is_valid_move(self.board, col, self.cols)
    
    def make_move(self, col: int) -> bool:
        """
        Make a move by dropping a piece in the specified column.
        
        Args:
            col (int): Column index (0-based)
            
        Returns:
            bool: True if the move was successful, False otherwise
        """
        if not self.is_valid_move(col):
            return False
        
        # Use JIT-compiled function to make the move
        row = _jit_make_move(self.board, col, self.current_player.value, self.rows)
        if row == -1:
            return False
        
        # Check for win or draw
        self._update_game_state(row, col)
        
        # Switch players if game is still in progress
        if self.game_state == GameState.IN_PROGRESS:
            self.current_player = Player.TWO if self.current_player == Player.ONE else Player.ONE
        
        return True
    
    def _update_game_state(self, last_row: int, last_col: int) -> None:
        """
        Update the game state after a move.
        
        Args:
            last_row (int): Row of the last placed piece
            last_col (int): Column of the last placed piece
        """
        # Check if current player won
        if self._check_win(last_row, last_col):
            if self.current_player == Player.ONE:
                self.game_state = GameState.PLAYER_ONE_WINS
            else:
                self.game_state = GameState.PLAYER_TWO_WINS
            return
        
        # Check for draw (board full)
        if not self.get_valid_moves():
            self.game_state = GameState.DRAW
    
    def _check_win(self, row: int, col: int) -> bool:
        """
        Check if the current player has won after placing a piece at (row, col).
        
        Args:
            row (int): Row of the placed piece
            col (int): Column of the placed piece
            
        Returns:
            bool: True if the current player has won, False otherwise
        """
        return _jit_check_win(self.board, row, col, self.current_player.value,
                             self.connect_length, self.rows, self.cols)
    
    def get_winner(self) -> Optional[Player]:
        """
        Get the winner of the game.
        
        Returns:
            Optional[Player]: The winning player, or None if no winner yet
        """
        if self.game_state == GameState.PLAYER_ONE_WINS:
            return Player.ONE
        elif self.game_state == GameState.PLAYER_TWO_WINS:
            return Player.TWO
        return None
    
    def is_game_over(self) -> bool:
        """
        Check if the game is over.
        
        Returns:
            bool: True if the game is over, False otherwise
        """
        return self.game_state != GameState.IN_PROGRESS
    
    def reset(self) -> None:
        """Reset the game to initial state."""
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = Player.ONE
        self.game_state = GameState.IN_PROGRESS
    
    def is_jit_enabled(self) -> bool:
        """
        Check if JIT compilation is enabled.
        
        Returns:
            bool: True if Numba JIT is available and enabled, False otherwise
        """
        return self._jit_enabled
    
    def __str__(self) -> str:
        """
        String representation of the board.
        
        Returns:
            str: Visual representation of the board
        """
        result = []
        
        # Column numbers
        col_numbers = " ".join(str(i % 10) for i in range(self.cols))
        result.append(f" {col_numbers}")
        result.append("+" + "-" * (2 * self.cols - 1) + "+")
        
        # Board rows
        for row in self.board:
            row_str = "|"
            for cell in row:
                if cell == 0:
                    row_str += " "
                elif cell == 1:
                    row_str += "X"
                else:
                    row_str += "O"
                row_str += "|"
            result.append(row_str)
        
        result.append("+" + "-" * (2 * self.cols - 1) + "+")
        
        # Game state info
        if self.game_state == GameState.IN_PROGRESS:
            result.append(f"Current player: {self.current_player.name}")
        elif self.game_state == GameState.DRAW:
            result.append("Game ended in a draw!")
        else:
            winner = self.get_winner()
            result.append(f"Player {winner.name} wins!")
        
        return "\n".join(result)