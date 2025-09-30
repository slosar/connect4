"""
Connect 4 Web Interface

A Flask web application for playing Connect 4 with configurable board sizes.
"""

import sys
import os
import random
from flask import Flask, render_template, request, jsonify, session
from typing import Dict, Any

# Add the src directory to the path so we can import the game engine
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from connect4 import Connect4, Player, GameState

app = Flask(__name__)
app.secret_key = 'connect4_secret_key_change_in_production'

def get_game() -> Connect4:
    """Get or create a game instance from the session."""
    if 'game_id' not in session or session['game_id'] not in games:
        # Create new game with default settings
        game = Connect4(rows=7, cols=7, connect_length=4)
        game_id = str(random.randint(100000, 999999))
        games[game_id] = game
        session['game_id'] = game_id
        # Only set default game mode if it's not already set
        if 'game_mode' not in session:
            session['game_mode'] = 'friend'
    return games[session['game_id']]

def serialize_game_state(game: Connect4) -> Dict[str, Any]:
    """Convert game state to JSON-serializable format."""
    return {
        'board': game.get_board(),
        'rows': game.rows,
        'cols': game.cols,
        'connect_length': game.connect_length,
        'current_player': game.current_player.value,
        'game_state': game.game_state.value,
        'valid_moves': game.get_valid_moves(),
        'winner': game.get_winner().value if game.get_winner() else None,
        'is_game_over': game.is_game_over()
    }

def make_computer_move(game: Connect4) -> bool:
    """Make a random move for the computer player."""
    valid_moves = game.get_valid_moves()
    if not valid_moves:
        return False
    
    # Simple random AI - pick a random valid column
    col = random.choice(valid_moves)
    return game.make_move(col)

# Global storage for game instances (in production, use a database)
games: Dict[str, Connect4] = {}

@app.route('/')
def index():
    """Main game page."""
    return render_template('index.html')

@app.route('/api/game/state')
def get_game_state():
    """Get current game state."""
    game = get_game()
    state = serialize_game_state(game)
    state['game_mode'] = session.get('game_mode', 'friend')
    return jsonify(state)

@app.route('/api/game/move', methods=['POST'])
def make_move():
    """Make a move in the game."""
    data = request.get_json()
    col = data.get('col')
    
    if col is None:
        return jsonify({'error': 'Column not specified'}), 400
    
    game = get_game()
    
    # Make the human player's move
    if not game.make_move(col):
        return jsonify({'error': 'Invalid move'}), 400
    
    response = serialize_game_state(game)
    
    # If playing against computer and game is still in progress, make computer move
    if (session.get('game_mode') == 'computer' and 
        not game.is_game_over() and 
        game.current_player == Player.TWO):
        
        if make_computer_move(game):
            response = serialize_game_state(game)
            response['computer_moved'] = True
    
    response['game_mode'] = session.get('game_mode', 'friend')
    return jsonify(response)

@app.route('/api/game/reset', methods=['POST'])
def reset_game():
    """Reset the current game."""
    game = get_game()
    game.reset()
    return jsonify(serialize_game_state(game))

@app.route('/api/game/new', methods=['POST'])
def new_game():
    """Create a new game with specified settings."""
    data = request.get_json()
    
    rows = data.get('rows', 7)
    cols = data.get('cols', 7)
    connect_length = data.get('connect_length', 4)
    game_mode = data.get('game_mode', 'friend')
    
    # Validate parameters
    if not (1 <= rows <= 100 and 1 <= cols <= 100):
        return jsonify({'error': 'Board dimensions must be between 1 and 100'}), 400
    
    if not (1 <= connect_length <= max(rows, cols)):
        return jsonify({'error': 'Connect length must be between 1 and max(rows, cols)'}), 400
    
    if game_mode not in ['friend', 'computer']:
        return jsonify({'error': 'Game mode must be "friend" or "computer"'}), 400
    
    try:
        # Create new game
        game = Connect4(rows=rows, cols=cols, connect_length=connect_length)
        game_id = str(random.randint(100000, 999999))
        games[game_id] = game
        session['game_id'] = game_id
        session['game_mode'] = game_mode
        
        response = serialize_game_state(game)
        response['game_mode'] = game_mode
        return jsonify(response)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)