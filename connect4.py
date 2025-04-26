import numpy as np
import random
import math
import sys
from typing import Tuple, List, Optional

# Game constants
ROWS = 6
COLS = 7
WINDOW_LENGTH = 4  # Number of pieces in a row to win
EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

class Connect4:
    def __init__(self):
        """Initialize the game board"""
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.game_over = False
        self.turn = 0  # Even for player, Odd for AIs
    
    def drop_piece(self, col: int, piece: int) -> bool:
        """Drop a piece into the specified column"""
        for row in range(ROWS-1, -1, -1):
            if self.board[row][col] == EMPTY:
                self.board[row][col] = piece
                return True
        return False
    
    def is_valid_location(self, col: int) -> bool:
        """Check if a column is valid for placing a piece"""
        return 0 <= col < COLS and self.board[0][col] == EMPTY
    
    def get_valid_locations(self) -> List[int]:
        """Get all valid columns where a piece can be dropped"""
        return [col for col in range(COLS) if self.is_valid_location(col)]
    
    def is_winning_move(self, piece: int) -> bool:
        """Check if the last move resulted in a win"""
        # Check horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                if (self.board[r][c] == piece and 
                    self.board[r][c+1] == piece and 
                    self.board[r][c+2] == piece and 
                    self.board[r][c+3] == piece):
                    return True
        
        # Check vertical
        for r in range(ROWS - 3):
            for c in range(COLS):
                if (self.board[r][c] == piece and 
                    self.board[r+1][c] == piece and 
                    self.board[r+2][c] == piece and 
                    self.board[r+3][c] == piece):
                    return True
        
        # Check positively sloped diagonals
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if (self.board[r][c] == piece and 
                    self.board[r+1][c+1] == piece and 
                    self.board[r+2][c+2] == piece and 
                    self.board[r+3][c+3] == piece):
                    return True
        
        # Check negatively sloped diagonals
        for r in range(3, ROWS):
            for c in range(COLS - 3):
                if (self.board[r][c] == piece and 
                    self.board[r-1][c+1] == piece and 
                    self.board[r-2][c+2] == piece and 
                    self.board[r-3][c+3] == piece):
                    return True
        
        return False
    
    def is_board_full(self) -> bool:
        """Check if the board is full (tie)"""
        return len(self.get_valid_locations()) == 0
    
    def evaluate_window(self, window: list, piece: int) -> int:
        """Score a window of 4 positions"""
        opponent_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE
        
        if window.count(piece) == 4:
            return 100
        elif window.count(piece) == 3 and window.count(EMPTY) == 1:
            return 5
        elif window.count(piece) == 2 and window.count(EMPTY) == 2:
            return 2
        
        if window.count(opponent_piece) == 3 and window.count(EMPTY) == 1:
            return -4
        
        return 0
    
    def score_position(self, piece: int) -> int:
        """Score the entire board position for the given piece"""
        score = 0
        
        # Score center column (preferable to control the center)
        center_array = [int(self.board[r][COLS//2]) for r in range(ROWS)]
        center_count = center_array.count(piece)
        score += center_count * 3
        
        # Score horizontal
        for r in range(ROWS):
            row_array = [int(self.board[r][c]) for c in range(COLS)]
            for c in range(COLS - 3):
                window = row_array[c:c+WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)
        
        # Score vertical
        for c in range(COLS):
            col_array = [int(self.board[r][c]) for r in range(ROWS)]
            for r in range(ROWS - 3):
                window = col_array[r:r+WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)
        
        # Score positive diagonal
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                window = [self.board[r+i][c+i] for i in range(WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)
        
        # Score negative diagonal
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                window = [self.board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)
        
        return score
    
    def is_terminal_node(self) -> bool:
        """Check if the game has ended"""
        return (self.is_winning_move(PLAYER_PIECE) or 
                self.is_winning_move(AI_PIECE) or 
                self.is_board_full())
    
    def minimax(self, depth: int, alpha: float, beta: float, maximizing_player: bool) -> Tuple[int, int]:
        """
        Minimax algorithm with alpha-beta pruning
        Returns (column, score)
        """
        valid_locations = self.get_valid_locations()
        
        # Terminal node (win/lose/tie or depth limit)
        if depth == 0 or self.is_terminal_node():
            if self.is_terminal_node():
                if self.is_winning_move(AI_PIECE):
                    return (None, 1000000)
                elif self.is_winning_move(PLAYER_PIECE):
                    return (None, -1000000)
                else:  # Game is over, no more valid moves
                    return (None, 0)
            else:  # Depth is zero
                return (None, self.score_position(AI_PIECE))
        
        if maximizing_player:
            value = -math.inf
            column = random.choice(valid_locations)
            for col in valid_locations:
                # Create a copy of the board
                board_copy = self.board.copy()
                # Try this move
                self.drop_piece(col, AI_PIECE)
                # Recursively call minimax
                new_score = self.minimax(depth-1, alpha, beta, False)[1]
                # Undo the move
                self.board = board_copy
                # Update best move if better
                if new_score > value:
                    value = new_score
                    column = col
                # Alpha-beta pruning
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return column, value
        
        else:  # Minimizing player
            value = math.inf
            column = random.choice(valid_locations)
            for col in valid_locations:
                # Create a copy of the board
                board_copy = self.board.copy()
                # Try this move
                self.drop_piece(col, PLAYER_PIECE)
                # Recursively call minimax
                new_score = self.minimax(depth-1, alpha, beta, True)[1]
                # Undo the move
                self.board = board_copy
                # Update best move if better
                if new_score < value:
                    value = new_score
                    column = col
                # Alpha-beta pruning
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return column, value
    
    def get_ai_move(self, difficulty: int = 5) -> int:
        """Get the best move for the AI using minimax with alpha-beta pruning"""
        # Difficulty level determines the depth of the search
        col, _ = self.minimax(depth=difficulty, alpha=-math.inf, beta=math.inf, maximizing_player=True)
        return col
    
    def print_board(self):
        """Print the current state of the board"""
        print("\n")
        print(" " + " ".join([str(i) for i in range(COLS)]))
        print("-" * (COLS * 2 - 1))
        
        for row in self.board:
            print("|", end="")
            for cell in row:
                if cell == EMPTY:
                    print(" ", end="|")
                elif cell == PLAYER_PIECE:
                    print("X", end="|")
                else:  # cell == AI_PIECE
                    print("O", end="|")
            print()
        print("-" * (COLS * 2 - 1))

def main():
    # Initialize the game
    game = Connect4()
    game_over = False
    
    # Choose difficulty
    print("Connect 4 - Player vs AI")
    print("Choose difficulty level (1-5):")
    try:
        difficulty = int(input("> "))
        if difficulty < 1 or difficulty > 5:
            difficulty = 3
            print("Invalid choice. Setting to medium difficulty (3).")
        else:
            levels = ["Very Easy", "Easy", "Medium", "Hard", "Very Hard"]
            print(f"Difficulty set to {levels[difficulty-1]}.")
    except ValueError:
        difficulty = 3
        print("Invalid input. Setting to medium difficulty (3).")
    
    # Main game loop
    while not game_over:
        # Print current board
        game.print_board()
        
        # Player's turn
        if game.turn % 2 == 0:
            print("Your turn (0-6):")
            try:
                col = int(input("> "))
                if not 0 <= col < COLS:
                    print("Invalid column. Please choose between 0 and 6.")
                    continue
                
                if not game.is_valid_location(col):
                    print("Column is full. Choose another one.")
                    continue
                
                game.drop_piece(col, PLAYER_PIECE)
                
                if game.is_winning_move(PLAYER_PIECE):
                    game.print_board()
                    print("You win!")
                    game_over = True
            except ValueError:
                print("Please enter a valid number between 0 and 6.")
                continue
        
        # AI's turn
        else:
            print("AI is thinking...")
            col = game.get_ai_move(difficulty)
            game.drop_piece(col, AI_PIECE)
            print(f"AI drops piece in column {col}")
            
            if game.is_winning_move(AI_PIECE):
                game.print_board()
                print("AI wins!")
                game_over = True
        
        # Check for tie
        if game.is_board_full():
            game.print_board()
            print("It's a tie!")
            game_over = True
        
        game.turn += 1
    
    print("Game Over!")
    play_again = input("Play again? (y/n): ").lower().strip()
    if play_again == 'y':
        main()

if __name__ == "__main__":
    main()