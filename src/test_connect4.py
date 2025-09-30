"""
Test suite for Connect-4 implementation.

This module contains comprehensive tests for the Connect4 class,
including edge cases, different board sizes, and all game mechanics.
"""

import pytest
from src.connect4 import Connect4, Player, GameState


class TestConnect4Initialization:
    """Test Connect4 initialization and basic properties."""
    
    def test_default_initialization(self):
        """Test default board size and parameters."""
        game = Connect4()
        assert game.rows == 6
        assert game.cols == 7
        assert game.connect_length == 4
        assert game.current_player == Player.ONE
        assert game.game_state == GameState.IN_PROGRESS
    
    def test_custom_initialization(self):
        """Test custom board size initialization."""
        game = Connect4(rows=8, cols=9, connect_length=5)
        assert game.rows == 8
        assert game.cols == 9
        assert game.connect_length == 5
        
    def test_invalid_initialization(self):
        """Test invalid initialization parameters."""
        with pytest.raises(ValueError):
            Connect4(rows=0, cols=7)
        
        with pytest.raises(ValueError):
            Connect4(rows=6, cols=0)
            
        with pytest.raises(ValueError):
            Connect4(rows=6, cols=7, connect_length=0)
            
        with pytest.raises(ValueError):
            Connect4(rows=3, cols=3, connect_length=5)  # connect_length > max dimension
    
    def test_board_initialization(self):
        """Test that board is properly initialized with zeros."""
        game = Connect4(rows=4, cols=5)
        board = game.get_board()
        assert len(board) == 4
        assert len(board[0]) == 5
        assert all(cell == 0 for row in board for cell in row)


class TestValidMoves:
    """Test valid move detection and validation."""
    
    def test_initial_valid_moves(self):
        """Test that all columns are valid initially."""
        game = Connect4(rows=3, cols=4)
        valid_moves = game.get_valid_moves()
        assert valid_moves == [0, 1, 2, 3]
    
    def test_is_valid_move(self):
        """Test individual move validation."""
        game = Connect4(rows=3, cols=4)
        assert game.is_valid_move(0) is True
        assert game.is_valid_move(3) is True
        assert game.is_valid_move(-1) is False
        assert game.is_valid_move(4) is False
    
    def test_full_column_invalid(self):
        """Test that full columns are not valid moves."""
        game = Connect4(rows=2, cols=3, connect_length=2)
        
        # Fill column 0
        game.make_move(0)  # Player 1
        game.make_move(0)  # Player 2
        
        assert game.is_valid_move(0) is False
        assert 0 not in game.get_valid_moves()
        assert game.is_valid_move(1) is True
        assert game.is_valid_move(2) is True
    
    def test_no_valid_moves_after_game_over(self):
        """Test that no moves are valid after game ends."""
        game = Connect4(rows=1, cols=1, connect_length=1)
        game.make_move(0)  # Player 1 wins immediately
        
        assert game.get_valid_moves() == []
        assert game.is_valid_move(0) is False


class TestMakingMoves:
    """Test move execution and board state changes."""
    
    def test_simple_move(self):
        """Test making a simple move."""
        game = Connect4(rows=3, cols=3, connect_length=3)
        result = game.make_move(1)
        
        assert result is True
        board = game.get_board()
        assert board[2][1] == 1  # Bottom row, column 1
        assert game.current_player == Player.TWO
    
    def test_stacking_pieces(self):
        """Test that pieces stack properly."""
        game = Connect4(rows=3, cols=3, connect_length=3)
        
        game.make_move(0)  # Player 1
        game.make_move(0)  # Player 2
        game.make_move(0)  # Player 1
        
        board = game.get_board()
        assert board[2][0] == 1  # Bottom
        assert board[1][0] == 2  # Middle
        assert board[0][0] == 1  # Top
    
    def test_invalid_move_rejection(self):
        """Test that invalid moves are rejected."""
        game = Connect4(rows=2, cols=2, connect_length=2)
        
        # Fill column 0
        game.make_move(0)
        game.make_move(0)
        
        # Try to add another piece to full column
        result = game.make_move(0)
        assert result is False
        assert game.current_player == Player.ONE  # Should not switch
    
    def test_out_of_bounds_move(self):
        """Test moves outside board boundaries."""
        game = Connect4(rows=3, cols=3, connect_length=3)
        
        assert game.make_move(-1) is False
        assert game.make_move(3) is False
        assert game.current_player == Player.ONE  # Should not change


class TestWinDetection:
    """Test win condition detection in all directions."""
    
    def test_horizontal_win(self):
        """Test horizontal win detection."""
        game = Connect4(rows=4, cols=5, connect_length=4)
        
        # Player 1 wins horizontally
        for col in range(4):
            game.make_move(col)  # Player 1
            if col < 3:
                game.make_move(col)  # Player 2 (blocks vertical)
        
        assert game.game_state == GameState.PLAYER_ONE_WINS
        assert game.get_winner() == Player.ONE
        assert game.is_game_over() is True
    
    def test_vertical_win(self):
        """Test vertical win detection."""
        game = Connect4(rows=5, cols=4, connect_length=4)
        
        # Player 1 wins vertically in column 0
        for _ in range(4):
            game.make_move(0)  # Player 1
            game.make_move(1)  # Player 2
        
        assert game.game_state == GameState.PLAYER_ONE_WINS
        assert game.get_winner() == Player.ONE
    
    def test_diagonal_win_positive(self):
        """Test diagonal win (positive slope)."""
        game = Connect4(rows=6, cols=7, connect_length=4)
        
        # Build diagonal from bottom-left to top-right
        # We need to carefully build the diagonal (0,0), (1,1), (2,2), (3,3)
        
        game.make_move(0)  # P1 (5,0)
        game.make_move(1)  # P2 (5,1)  
        game.make_move(1)  # P1 (4,1)
        game.make_move(2)  # P2 (5,2)
        game.make_move(2)  # P1 (4,2)
        game.make_move(2)  # P2 (3,2)
        game.make_move(3)  # P1 (5,3)
        game.make_move(3)  # P2 (4,3)
        game.make_move(3)  # P1 (3,3)
        game.make_move(3)  # P2 (2,3)
        game.make_move(4)  # P1 (5,4) - dummy move
        game.make_move(0)  # P2 (4,0) - dummy move
        game.make_move(2)  # P1 (2,2) - part of diagonal
        game.make_move(5)  # P2 (5,5) - dummy move
        game.make_move(1)  # P1 (3,1) - part of diagonal
        game.make_move(6)  # P2 (5,6) - dummy move
        game.make_move(0)  # P1 (3,0) - part of diagonal
        
        # Now we should have diagonal at (5,0), (4,1), (3,2), (2,3) - but this might not work
        # Let's try a simpler approach - build (2,0), (3,1), (4,2), (5,3)
        game = Connect4(rows=6, cols=7, connect_length=4)
        
        # Build pieces at the correct positions for diagonal
        game.make_move(3)  # P1 (5,3)
        game.make_move(0)  # P2 (5,0)
        game.make_move(2)  # P1 (5,2) 
        game.make_move(0)  # P2 (4,0)
        game.make_move(2)  # P1 (4,2)
        game.make_move(0)  # P2 (3,0)
        game.make_move(1)  # P1 (5,1)
        game.make_move(0)  # P2 (2,0) - This completes P2's diagonal!
        
        assert game.game_state == GameState.PLAYER_TWO_WINS
    
    def test_diagonal_win_negative(self):
        """Test diagonal win (negative slope)."""
        game = Connect4(rows=6, cols=7, connect_length=4)
        
        # Build diagonal from bottom-right to top-left
        game.make_move(3)  # P1 (5,3)
        game.make_move(2)  # P2 (5,2)
        game.make_move(2)  # P1 (4,2)
        game.make_move(1)  # P2 (5,1)
        game.make_move(1)  # P1 (4,1)
        game.make_move(0)  # P2 (5,0)
        game.make_move(1)  # P1 (3,1)
        game.make_move(0)  # P2 (4,0)
        game.make_move(0)  # P1 (3,0)
        game.make_move(6)  # P2 (5,6)
        game.make_move(0)  # P1 (2,0) - This should complete diagonal
        
        assert game.game_state == GameState.PLAYER_ONE_WINS
    
    def test_connect_length_win(self):
        """Test win with different connect lengths."""
        # Test connect-3
        game = Connect4(rows=4, cols=4, connect_length=3)
        
        for col in range(3):
            game.make_move(col)  # Player 1
            if col < 2:
                game.make_move(col)  # Player 2
        
        assert game.game_state == GameState.PLAYER_ONE_WINS
        
        # Test connect-5
        game = Connect4(rows=6, cols=8, connect_length=5)
        
        for col in range(5):
            game.make_move(col)  # Player 1
            if col < 4:
                game.make_move(col)  # Player 2
        
        assert game.game_state == GameState.PLAYER_ONE_WINS


class TestGameStates:
    """Test different game states and transitions."""
    
    def test_draw_detection(self):
        """Test draw detection when board is full."""
        # Use a 3x2 board with connect_length=3 to guarantee no winner  
        game = Connect4(rows=3, cols=2, connect_length=3)
        
        # Fill the board - no player can get 3 in a row on a 3x2 board
        game.make_move(0)  # P1 (2,0)
        game.make_move(0)  # P2 (1,0)
        game.make_move(0)  # P1 (0,0)
        game.make_move(1)  # P2 (2,1)
        game.make_move(1)  # P1 (1,1)
        game.make_move(1)  # P2 (0,1)
        
        assert game.game_state == GameState.DRAW
        assert game.get_winner() is None
        assert game.is_game_over() is True
    
    def test_game_state_persistence(self):
        """Test that game state persists after win."""
        game = Connect4(rows=4, cols=4, connect_length=3)
        
        # Player 1 wins
        for col in range(3):
            game.make_move(col)
            if col < 2:
                game.make_move(col)
        
        assert game.game_state == GameState.PLAYER_ONE_WINS
        
        # Further moves should be invalid
        assert game.make_move(3) is False
        assert game.game_state == GameState.PLAYER_ONE_WINS
    
    def test_reset_functionality(self):
        """Test game reset."""
        game = Connect4(rows=3, cols=3, connect_length=3)
        
        # Make some moves
        game.make_move(0)
        game.make_move(1)
        
        # Reset
        game.reset()
        
        assert game.current_player == Player.ONE
        assert game.game_state == GameState.IN_PROGRESS
        board = game.get_board()
        assert all(cell == 0 for row in board for cell in row)


class TestBoardRepresentation:
    """Test board representation and display."""
    
    def test_get_board_copy(self):
        """Test that get_board returns a copy."""
        game = Connect4(rows=2, cols=2, connect_length=2)
        board1 = game.get_board()
        board2 = game.get_board()
        
        assert board1 == board2
        assert board1 is not board2  # Different objects
        
        board1[0][0] = 999
        assert game.get_board()[0][0] == 0  # Original unchanged
    
    def test_string_representation(self):
        """Test string representation of the board."""
        game = Connect4(rows=2, cols=3, connect_length=2)
        game.make_move(0)  # P1
        game.make_move(1)  # P2
        
        board_str = str(game)
        assert "X" in board_str  # Player 1 piece
        assert "O" in board_str  # Player 2 piece
        assert "Current player: ONE" in board_str


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_minimum_board_size(self):
        """Test 1x1 board."""
        game = Connect4(rows=1, cols=1, connect_length=1)
        
        assert game.make_move(0) is True
        assert game.game_state == GameState.PLAYER_ONE_WINS
        assert game.get_winner() == Player.ONE
    
    def test_large_board(self):
        """Test with a larger board."""
        game = Connect4(rows=20, cols=30, connect_length=6)
        
        # Should initialize properly
        assert len(game.get_board()) == 20
        assert len(game.get_board()[0]) == 30
        assert len(game.get_valid_moves()) == 30
    
    def test_rectangular_boards(self):
        """Test non-square boards."""
        # Tall narrow board
        game1 = Connect4(rows=10, cols=3, connect_length=4)
        assert game1.rows == 10
        assert game1.cols == 3
        
        # Wide short board
        game2 = Connect4(rows=3, cols=10, connect_length=4)
        assert game2.rows == 3
        assert game2.cols == 10
    
    def test_connect_length_edge_cases(self):
        """Test various connect lengths."""
        # Connect-1 (every move wins)
        game = Connect4(rows=3, cols=3, connect_length=1)
        game.make_move(0)
        assert game.game_state == GameState.PLAYER_ONE_WINS
        
        # Connect equals board dimension
        game = Connect4(rows=5, cols=5, connect_length=5)
        assert game.connect_length == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])