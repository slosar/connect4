/**
 * Connect 4 Web Game JavaScript
 * Handles game interactions, API calls, and UI updates
 */

class Connect4Game {
    constructor() {
        this.gameState = null;
        this.isPlayerTurn = true;
        this.init();
    }

    init() {
        this.bindEvents();
        this.loadGameState();
    }

    bindEvents() {
        // Game control buttons
        document.getElementById('new-game-btn').addEventListener('click', () => this.showSettings());
        document.getElementById('reset-btn').addEventListener('click', () => this.resetGame());
        document.getElementById('settings-btn').addEventListener('click', () => this.showSettings());

        // Modal controls
        document.querySelector('.close').addEventListener('click', () => this.hideModal('settings-modal'));
        document.getElementById('cancel-settings').addEventListener('click', () => this.hideModal('settings-modal'));
        document.getElementById('apply-settings').addEventListener('click', () => this.applySettings());

        // Winner modal controls
        document.getElementById('play-again').addEventListener('click', () => this.resetGame());
        document.getElementById('new-game-winner').addEventListener('click', () => {
            this.hideModal('winner-modal');
            this.showSettings();
        });

        // Error message close
        document.getElementById('close-error').addEventListener('click', () => this.hideError());

        // Click outside modal to close
        window.addEventListener('click', (event) => {
            if (event.target.classList.contains('modal')) {
                this.hideModal(event.target.id);
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') {
                this.hideModal('settings-modal');
                this.hideModal('winner-modal');
            } else if (event.key >= '1' && event.key <= '9') {
                const col = parseInt(event.key) - 1;
                if (this.gameState && col < this.gameState.cols) {
                    this.makeMove(col);
                }
            }
        });
    }

    async loadGameState() {
        try {
            const response = await fetch('/api/game/state');
            if (!response.ok) throw new Error('Failed to load game state');
            
            this.gameState = await response.json();
            
            // Ensure game_mode is always set to a valid value
            if (!this.gameState.game_mode) {
                this.gameState.game_mode = 'friend';
            }
            
            // Set initial isPlayerTurn based on game mode and current player
            if (this.gameState.game_mode === 'friend') {
                this.isPlayerTurn = true;
            } else if (this.gameState.game_mode === 'computer') {
                this.isPlayerTurn = (this.gameState.current_player === 1);
            }
            
            this.updateUI();
        } catch (error) {
            this.showError('Failed to load game: ' + error.message);
        }
    }

    async makeMove(col) {
        if (!this.isPlayerTurn || !this.gameState || this.gameState.is_game_over) {
            return;
        }

        // Check if column is valid
        if (!this.gameState.valid_moves.includes(col)) {
            this.showError('Invalid move: Column is full');
            return;
        }

        this.showLoading(true);
        this.isPlayerTurn = false;

        try {
            const response = await fetch('/api/game/move', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ col: col })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to make move');
            }

            this.gameState = await response.json();
            
            // Set isPlayerTurn BEFORE updateUI so board clicks work correctly
            if (this.gameState.is_game_over) {
                // Game over - no more moves allowed
                this.isPlayerTurn = false;
            } else {
                // In friend mode, always allow moves after each move
                // In computer mode, allow moves when it's player 1's turn (current_player === 1)
                if (this.gameState.game_mode === 'friend') {
                    this.isPlayerTurn = true;
                } else if (this.gameState.game_mode === 'computer') {
                    // In computer mode, player can move when it's their turn (player 1)
                    this.isPlayerTurn = (this.gameState.current_player === 1);
                }
            }
            
            this.updateUI();

            // Check if game is over
            if (this.gameState.is_game_over) {
                setTimeout(() => this.showWinner(), 500);
            }

        } catch (error) {
            this.showError(error.message);
            this.isPlayerTurn = true;
        } finally {
            this.showLoading(false);
        }
    }

    async resetGame() {
        this.showLoading(true);
        
        try {
            const response = await fetch('/api/game/reset', { method: 'POST' });
            if (!response.ok) throw new Error('Failed to reset game');
            
            this.gameState = await response.json();
            
            // Reset isPlayerTurn based on game mode
            if (this.gameState.game_mode === 'friend') {
                this.isPlayerTurn = true;
            } else if (this.gameState.game_mode === 'computer') {
                this.isPlayerTurn = (this.gameState.current_player === 1);
            }
            
            this.updateUI();
            this.hideModal('winner-modal');
        } catch (error) {
            this.showError('Failed to reset game: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    async applySettings() {
        const rows = parseInt(document.getElementById('board-rows').value);
        const cols = parseInt(document.getElementById('board-cols').value);
        const connectLength = parseInt(document.getElementById('connect-length').value);
        const gameMode = document.getElementById('game-mode').value;

        // Validate settings
        if (rows < 3 || rows > 100 || cols < 3 || cols > 100) {
            this.showError('Board dimensions must be between 3 and 100');
            return;
        }

        if (connectLength < 3 || connectLength > Math.max(rows, cols)) {
            this.showError('Connect length must be between 3 and max(rows, cols)');
            return;
        }

        this.showLoading(true);

        try {
            const response = await fetch('/api/game/new', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    rows: rows,
                    cols: cols,
                    connect_length: connectLength,
                    game_mode: gameMode
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to create new game');
            }

            this.gameState = await response.json();
            
            // Set isPlayerTurn based on new game mode
            if (this.gameState.game_mode === 'friend') {
                this.isPlayerTurn = true;
            } else if (this.gameState.game_mode === 'computer') {
                this.isPlayerTurn = (this.gameState.current_player === 1);
            }
            
            this.updateUI();
            this.hideModal('settings-modal');
            this.hideModal('winner-modal');

        } catch (error) {
            this.showError(error.message);
        } finally {
            this.showLoading(false);
        }
    }

    updateUI() {
        if (!this.gameState) return;

        this.updateBoard();
        this.updateGameInfo();
        this.updatePlayerIndicator();
    }

    updateBoard() {
        const boardElement = document.getElementById('game-board');
        boardElement.innerHTML = '';

        // Create board grid
        const boardGrid = document.createElement('div');
        boardGrid.className = 'board-grid';
        boardGrid.style.gridTemplateColumns = `repeat(${this.gameState.cols}, 1fr)`;
        boardGrid.style.gridTemplateRows = `repeat(${this.gameState.rows}, 1fr)`;

        // Create columns for hover effect
        const columns = [];
        for (let col = 0; col < this.gameState.cols; col++) {
            const columnDiv = document.createElement('div');
            columnDiv.className = 'board-column';
            columnDiv.style.gridColumn = col + 1;
            columnDiv.style.gridRow = `1 / ${this.gameState.rows + 1}`;
            columnDiv.style.display = 'flex';
            columnDiv.style.flexDirection = 'column';
            columnDiv.style.gap = '8px';
            columns.push(columnDiv);
        }

        // Create cells
        for (let row = 0; row < this.gameState.rows; row++) {
            for (let col = 0; col < this.gameState.cols; col++) {
                const cell = document.createElement('div');
                cell.className = 'board-cell';
                cell.dataset.row = row;
                cell.dataset.col = col;

                const cellValue = this.gameState.board[row][col];
                if (cellValue === 1) {
                    cell.classList.add('player1');
                } else if (cellValue === 2) {
                    cell.classList.add('player2');
                }

                // Add click handler only for empty cells in valid columns
                const canClick = cellValue === 0 && this.gameState.valid_moves.includes(col) && 
                    !this.gameState.is_game_over && 
                    (this.gameState.game_mode === 'friend' || 
                     (this.gameState.game_mode === 'computer' && this.isPlayerTurn));
                
                if (canClick) {
                    cell.addEventListener('click', () => this.makeMove(col));
                } else if (cellValue === 0) {
                    cell.classList.add('disabled');
                }

                columns[col].appendChild(cell);
            }
        }

        // Add columns to board
        columns.forEach(column => boardGrid.appendChild(column));
        boardElement.appendChild(boardGrid);
    }

    updateGameInfo() {
        document.getElementById('board-size').textContent = 
            `${this.gameState.rows}×${this.gameState.cols}`;
        document.getElementById('connect-display').textContent = 
            this.gameState.connect_length;
        
        const gameModeDisplay = document.getElementById('game-mode-display');
        gameModeDisplay.textContent = this.gameState.game_mode === 'computer' ? 
            'Playing against Computer' : 'Playing with Friend';
    }

    updatePlayerIndicator() {
        const playerIndicator = document.getElementById('current-player');
        
        if (this.gameState.is_game_over) {
            if (this.gameState.winner) {
                playerIndicator.textContent = `Player ${this.gameState.winner} Wins!`;
                playerIndicator.className = `player-indicator player${this.gameState.winner}`;
            } else {
                playerIndicator.textContent = "It's a Draw!";
                playerIndicator.className = 'player-indicator';
            }
        } else {
            const currentPlayer = this.gameState.current_player;
            playerIndicator.textContent = `Player ${currentPlayer}'s Turn`;
            playerIndicator.className = `player-indicator player${currentPlayer}`;
            
            // Show waiting message for computer turn
            if (this.gameState.game_mode === 'computer' && currentPlayer === 2 && !this.isPlayerTurn) {
                playerIndicator.textContent = 'Computer is thinking...';
            }
        }
    }

    showSettings() {
        // Populate current settings
        document.getElementById('board-rows').value = this.gameState ? this.gameState.rows : 7;
        document.getElementById('board-cols').value = this.gameState ? this.gameState.cols : 7;
        document.getElementById('connect-length').value = this.gameState ? this.gameState.connect_length : 4;
        document.getElementById('game-mode').value = this.gameState ? this.gameState.game_mode : 'friend';
        
        this.showModal('settings-modal');
    }

    showWinner() {
        const winnerModal = document.getElementById('winner-modal');
        const winnerTitle = document.getElementById('winner-title');
        const winnerMessage = document.getElementById('winner-message');

        if (this.gameState.winner) {
            winnerTitle.textContent = 'We have a winner!';
            winnerMessage.textContent = `Player ${this.gameState.winner} Wins!`;
            winnerMessage.className = `winner-message player${this.gameState.winner}`;
        } else {
            winnerTitle.textContent = 'Game Over!';
            winnerMessage.textContent = "It's a Draw!";
            winnerMessage.className = 'winner-message draw';
        }

        this.showModal('winner-modal');
    }

    showModal(modalId) {
        document.getElementById(modalId).style.display = 'block';
    }

    hideModal(modalId) {
        document.getElementById(modalId).style.display = 'none';
    }

    showError(message) {
        document.getElementById('error-text').textContent = message;
        document.getElementById('error-message').classList.remove('hidden');
        
        // Auto-hide after 5 seconds
        setTimeout(() => this.hideError(), 5000);
    }

    hideError() {
        document.getElementById('error-message').classList.add('hidden');
    }

    showLoading(show) {
        if (show) {
            document.getElementById('loading').classList.remove('hidden');
        } else {
            document.getElementById('loading').classList.add('hidden');
        }
    }
}

// Initialize the game when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new Connect4Game();
});