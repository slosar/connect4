# Connect-4 Game Implementation

A flexible, well-tested Connect-4 implementation in Python that supports arbitrary board sizes and connection lengths.

## Features

- **Arbitrary Board Sizes**: Play on any size board (rows × columns)
- **Configurable Win Condition**: Set how many pieces in a row are needed to win
- **Complete Game Logic**: Handles all game states, win detection, and move validation
- **Comprehensive Testing**: Full test suite with edge cases and boundary conditions
- **Clean API**: Easy-to-use object-oriented interface
- **Visual Display**: ASCII representation of the game board

## Installation

No external dependencies are required for the core game. For running tests, you'll need pytest:

```bash
pip install pytest
```

## Quick Start

```python
from src.connect4 import Connect4

# Create a standard 6×7 Connect-4 game
game = Connect4()

# Make some moves
game.make_move(3)  # Player 1 drops in column 3
game.make_move(3)  # Player 2 drops in column 3
game.make_move(2)  # Player 1 drops in column 2

# Check game state
print(game)  # Display the board
print(f"Current player: {game.current_player.name}")
print(f"Valid moves: {game.get_valid_moves()}")
print(f"Game over: {game.is_game_over()}")

# Custom board size with different win condition
custom_game = Connect4(rows=5, cols=8, connect_length=5)
```

## API Reference

### Class: Connect4

#### Constructor
```python
Connect4(rows=6, cols=7, connect_length=4)
```

- `rows` (int): Number of rows in the board (default: 6)
- `cols` (int): Number of columns in the board (default: 7)  
- `connect_length` (int): Number of pieces needed in a row to win (default: 4)

#### Methods

##### Game State
- `get_board() -> List[List[int]]`: Returns a copy of the current board state
- `is_game_over() -> bool`: Check if the game has ended
- `get_winner() -> Optional[Player]`: Get the winning player (if any)
- `reset() -> None`: Reset the game to initial state

##### Making Moves
- `make_move(col: int) -> bool`: Drop a piece in the specified column
- `is_valid_move(col: int) -> bool`: Check if a move is valid
- `get_valid_moves() -> List[int]`: Get all valid column indices

##### Display
- `__str__() -> str`: ASCII representation of the board

#### Properties
- `rows`: Number of rows
- `cols`: Number of columns  
- `connect_length`: Number of pieces needed to win
- `current_player`: Current player (Player.ONE or Player.TWO)
- `game_state`: Current game state (GameState enum)

### Enums

#### Player
- `Player.ONE`: First player (represented as 'X' in display)
- `Player.TWO`: Second player (represented as 'O' in display)

#### GameState
- `GameState.IN_PROGRESS`: Game is still being played
- `GameState.PLAYER_ONE_WINS`: Player 1 has won
- `GameState.PLAYER_TWO_WINS`: Player 2 has won
- `GameState.DRAW`: Game ended in a draw (board full, no winner)

## Examples

### Basic Game
```python
from src.connect4 import Connect4, Player

game = Connect4()

# Play until someone wins or board is full
while not game.is_game_over():
    valid_moves = game.get_valid_moves()
    col = int(input(f"Player {game.current_player.name}, choose column {valid_moves}: "))
    
    if game.make_move(col):
        print(game)
    else:
        print("Invalid move!")

if game.get_winner():
    print(f"Player {game.get_winner().name} wins!")
else:
    print("It's a draw!")
```

### Custom Board Size
```python
# Create a smaller board with connect-3
game = Connect4(rows=4, cols=5, connect_length=3)

# Quick vertical win
game.make_move(2)  # P1
game.make_move(1)  # P2  
game.make_move(2)  # P1
game.make_move(1)  # P2
game.make_move(2)  # P1 wins!

print(game.get_winner())  # Player.ONE
```

### Edge Cases
```python
# Minimum possible game
tiny_game = Connect4(rows=1, cols=1, connect_length=1)
tiny_game.make_move(0)  # Instant win!

# Large board
large_game = Connect4(rows=20, cols=30, connect_length=6)

# Impossible to win scenario
draw_game = Connect4(rows=2, cols=2, connect_length=4)
# Fill all 4 spaces - will result in draw
```

## Running the Examples

```bash
python example.py
```

This will run through various demonstrations including:
- Basic gameplay
- Custom board sizes
- Edge cases
- Game state analysis
- Interactive play (commented out by default)

## Running Tests

```bash
# Run all tests
pytest src/test_connect4.py -v

# Run specific test class
pytest src/test_connect4.py::TestWinDetection -v

# Run with coverage (if pytest-cov is installed)
pytest src/test_connect4.py --cov=src.connect4
```

### Test Coverage

The test suite includes:
- **Initialization tests**: Valid/invalid parameters, board setup
- **Move validation**: Valid moves, invalid moves, full columns
- **Move execution**: Piece placement, stacking, player switching
- **Win detection**: All directions (horizontal, vertical, both diagonals)
- **Game states**: In progress, wins, draws, game over conditions
- **Edge cases**: Minimum/maximum sizes, different connect lengths
- **Board representation**: Display formatting, board copying

## Architecture

The implementation uses clean object-oriented design with:

- **Separation of concerns**: Game logic, state management, and display are well separated
- **Immutable external interface**: `get_board()` returns copies to prevent external modification
- **Enum-based states**: Type-safe player and game state representations
- **Comprehensive validation**: All inputs are validated with clear error messages
- **Flexible design**: Supports any reasonable board size and win condition

## Board Representation

- Board positions are stored as integers: 0 (empty), 1 (Player 1), 2 (Player 2)
- Coordinates are (row, col) with (0,0) at top-left
- Pieces "fall" to the lowest available position in each column
- Win detection checks all four directions from each placed piece

## Error Handling

The implementation includes robust error handling:
- Invalid board dimensions raise `ValueError`
- Invalid moves return `False` (don't raise exceptions)
- Out-of-bounds moves are safely rejected
- Game state prevents moves after game ends

## Performance

- Move validation: O(1)
- Move execution: O(rows) to find lowest position
- Win detection: O(connect_length) for each direction
- Overall complexity is very reasonable for typical board sizes

## Contributing

When contributing:
1. Maintain test coverage for new features
2. Follow the existing code style
3. Update documentation for API changes
4. Test with various board sizes and edge cases