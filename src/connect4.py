"""
Connect-4 Game Implementation

A flexible Connect-4 implementation that supports arbitrary board sizes.
Players take turns dropping pieces into columns, with the goal of connecting
4 pieces in a row (horizontally, vertically, or diagonally).
"""

from typing import List, Tuple, Optional, Literal
from enum import Enum


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
            
        self.rows = rows
        self.cols = cols
        self.connect_length = connect_length
        self.board = [[0 for _ in range(cols)] for _ in range(rows)]
        self.current_player = Player.ONE
        self.game_state = GameState.IN_PROGRESS
        
    def get_board(self) -> List[List[int]]:
        """
        Get a copy of the current board state.
        
        Returns:
            List[List[int]]: A copy of the board
        """
        return [row[:] for row in self.board]
    
    def get_valid_moves(self) -> List[int]:
        """
        Get all valid column indices where a piece can be dropped.
        
        Returns:
            List[int]: List of valid column indices (0-based)
        """
        if self.game_state != GameState.IN_PROGRESS:
            return []
        
        valid_moves = []
        for col in range(self.cols):
            if self.board[0][col] == 0:  # Top row is empty
                valid_moves.append(col)
        return valid_moves
    
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
        if col < 0 or col >= self.cols:
            return False
        return self.board[0][col] == 0
    
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
        
        # Find the lowest empty row in the column
        for row in range(self.rows - 1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = self.current_player.value
                break
        
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
        player_value = self.current_player.value
        
        # Check all four directions: horizontal, vertical, diagonal
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal /
            (1, -1),  # Diagonal \
        ]
        
        for dr, dc in directions:
            count = 1  # Count the piece just placed
            
            # Check in positive direction
            r, c = row + dr, col + dc
            while (0 <= r < self.rows and 0 <= c < self.cols and 
                   self.board[r][c] == player_value):
                count += 1
                r, c = r + dr, c + dc
            
            # Check in negative direction
            r, c = row - dr, col - dc
            while (0 <= r < self.rows and 0 <= c < self.cols and 
                   self.board[r][c] == player_value):
                count += 1
                r, c = r - dr, c - dc
            
            if count >= self.connect_length:
                return True
        
        return False
    
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
        self.board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.current_player = Player.ONE
        self.game_state = GameState.IN_PROGRESS
    
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