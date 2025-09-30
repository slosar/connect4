#!/usr/bin/env python3
"""
Example usage of the Connect-4 implementation.

This script demonstrates how to use the Connect4 class to play a game,
including different board sizes and game scenarios.
"""

from src.connect4 import Connect4, Player, GameState


def example_basic_game():
    """Demonstrate a basic Connect-4 game."""
    print("=== Basic Connect-4 Game ===")
    
    # Create a standard 6x7 board
    game = Connect4()
    print("Created a standard 6x7 Connect-4 board")
    print(game)
    print()
    
    # Play some moves
    moves = [3, 3, 2, 4, 1, 5, 0]  # Player 1 will win
    for i, col in enumerate(moves):
        player = "Player 1" if i % 2 == 0 else "Player 2"
        print(f"{player} plays column {col}")
        
        if game.make_move(col):
            print(game)
            print()
            
            if game.is_game_over():
                break
        else:
            print(f"Invalid move: column {col}")
    
    print(f"Game result: {game.game_state.value}")
    if game.get_winner():
        print(f"Winner: {game.get_winner().name}")
    print("\n" + "="*50 + "\n")


def example_custom_board():
    """Demonstrate Connect-4 with custom board size."""
    print("=== Custom Board Size (4x5, Connect-3) ===")
    
    # Create a smaller board with connect-3
    game = Connect4(rows=4, cols=5, connect_length=3)
    print("Created a 4x5 board with connect-3 to win")
    print(game)
    print()
    
    # Quick game
    moves = [2, 1, 2, 3, 2]  # Player 1 wins vertically
    for i, col in enumerate(moves):
        player = "Player 1" if i % 2 == 0 else "Player 2"
        print(f"{player} plays column {col}")
        
        game.make_move(col)
        print(game)
        print()
        
        if game.is_game_over():
            break
    
    print("\n" + "="*50 + "\n")


def example_interactive_game():
    """Demonstrate an interactive game."""
    print("=== Interactive Connect-4 ===")
    print("Enter column numbers (0-based) to play")
    print("Enter 'q' to quit")
    
    game = Connect4()
    print(game)
    
    while not game.is_game_over():
        current_player = "Player 1" if game.current_player == Player.ONE else "Player 2"
        valid_moves = game.get_valid_moves()
        
        print(f"\n{current_player}'s turn")
        print(f"Valid moves: {valid_moves}")
        
        try:
            user_input = input("Enter column: ").strip()
            if user_input.lower() == 'q':
                print("Game quit by user")
                return
            
            col = int(user_input)
            
            if game.make_move(col):
                print(game)
            else:
                print(f"Invalid move: {col}")
                
        except (ValueError, KeyboardInterrupt):
            print("Invalid input or interrupted. Exiting...")
            return
    
    print(f"\nGame Over! Result: {game.game_state.value}")
    if game.get_winner():
        winner = "Player 1" if game.get_winner() == Player.ONE else "Player 2"
        print(f"Congratulations {winner}!")


def example_edge_cases():
    """Demonstrate edge cases and special scenarios."""
    print("=== Edge Cases and Special Scenarios ===")
    
    # Tiny board
    print("1. Tiny 2x2 board (connect-2, will draw):")
    tiny_game = Connect4(rows=2, cols=2, connect_length=2)
    
    # Fill the board (will result in draw)
    moves = [0, 0, 1, 1]
    for col in moves:
        tiny_game.make_move(col)
    
    print(tiny_game)
    print(f"Result: {tiny_game.game_state.value}\n")
    
    # Instant win board
    print("2. Instant win (1x1 board, connect-1):")
    instant_game = Connect4(rows=1, cols=1, connect_length=1)
    instant_game.make_move(0)
    print(instant_game)
    print()
    
    # Large board
    print("3. Large 10x15 board with connect-5:")
    large_game = Connect4(rows=10, cols=15, connect_length=5)
    print(f"Board size: {large_game.rows}x{large_game.cols}")
    print(f"Connect length: {large_game.connect_length}")
    print(f"Valid moves: {len(large_game.get_valid_moves())}")
    print()


def example_game_analysis():
    """Demonstrate game state analysis."""
    print("=== Game State Analysis ===")
    
    game = Connect4(rows=5, cols=6, connect_length=4)
    
    print("Initial state:")
    print(f"Current player: {game.current_player.name}")
    print(f"Game state: {game.game_state.value}")
    print(f"Valid moves: {game.get_valid_moves()}")
    print(f"Is game over: {game.is_game_over()}")
    print(f"Winner: {game.get_winner()}")
    print()
    
    # Make some moves
    moves = [2, 2, 2, 3, 3, 4]
    for col in moves:
        game.make_move(col)
    
    print("After some moves:")
    print(game)
    print(f"Current player: {game.current_player.name}")
    print(f"Valid moves: {game.get_valid_moves()}")
    print(f"Number of valid moves: {len(game.get_valid_moves())}")
    
    # Show board state as 2D array
    board = game.get_board()
    print("\nBoard as 2D array:")
    for row in board:
        print(row)


def main():
    """Run all examples."""
    print("Connect-4 Implementation Examples")
    print("=" * 50)
    print()
    
    try:
        example_basic_game()
        example_custom_board()
        example_edge_cases()
        example_game_analysis()
        
        # Uncomment the line below for interactive play
        # example_interactive_game()
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()