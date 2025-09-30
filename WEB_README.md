# Connect 4 Web Interface

A web-based Connect 4 game with configurable board sizes and game modes.

## Features

✅ **Play Modes**
- Play with a friend (local multiplayer)
- Play against computer (random AI)

✅ **Configurable Settings**
- Board dimensions: 3×3 to 100×100
- Connect length: 3 to max(rows, cols)
- Default: 7×7 board with Connect 4

✅ **Interactive Gameplay**
- Click on columns to drop pieces
- Visual board with animated pieces
- Real-time game state updates
- Winner detection and announcements

✅ **Game Controls**
- Reset game mid-play
- Start new games with different settings
- Responsive design for all screen sizes

## How to Run

1. **Install Dependencies**
   ```bash
   pip install flask
   ```

2. **Start the Web Server**
   ```bash
   cd web
   python app.py
   ```

3. **Open in Browser**
   Navigate to: `http://localhost:5000`

## How to Play

1. **Choose Game Mode**: Click "Settings" to select:
   - Play with Friend: Take turns on the same device
   - Play against Computer: Computer makes random moves

2. **Configure Board**: Set your preferred:
   - Board size (rows × columns)
   - Number of pieces needed to connect to win

3. **Make Moves**: 
   - Click on any column to drop a piece
   - Player 1 (red) always goes first
   - Pieces fall to the lowest available spot

4. **Win Conditions**:
   - Connect the required number of pieces in a row
   - Can be horizontal, vertical, or diagonal
   - Game declares winner automatically

5. **Game Controls**:
   - **Reset**: Clear the board and start over with same settings
   - **New Game**: Change settings and start fresh
   - **Settings**: Modify game configuration anytime

## Keyboard Shortcuts

- **1-9**: Drop piece in column (if valid)
- **Escape**: Close modal dialogs

## Technical Details

- **Backend**: Flask (Python)
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Game Engine**: Custom Connect 4 implementation in `src/connect4.py`
- **AI**: Simple random move selection for computer player

## File Structure

```
web/
├── app.py              # Flask application
├── templates/
│   └── index.html      # Main game interface
└── static/
    ├── style.css       # Game styling
    └── script.js       # Game logic and interactions
```

## API Endpoints

- `GET /` - Main game page
- `GET /api/game/state` - Get current game state
- `POST /api/game/move` - Make a move
- `POST /api/game/reset` - Reset current game
- `POST /api/game/new` - Create new game with settings

## Future Enhancements

- Smarter AI opponents
- Online multiplayer
- Game statistics and history
- Custom piece colors and themes
- Save/load game states