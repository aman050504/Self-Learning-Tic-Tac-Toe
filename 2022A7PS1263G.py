import sys
import random
import numpy as np
import json
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from TicTacToe import *
import os
"""
You may import additional, commonly used libraries that are widely installed.
Please do not request the installation of new libraries to run your program.
"""

class PlayerSQN:
    def __init__(self, model_file='2022A7PS1263G_MODEL.h5'):
        """
        Initializes the PlayerSQN class.
        """
        print("Initializing Agent...")
        self.state_size = 9  
        self.action_size = 9  
        self.replay_buffer = deque(maxlen=10000) 
        self.GAMMA = 0.95    
        self.epsilon = 1.0   
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "2022A7PS1263G_MODEL.h5")
        
        self.model = self.load_model(model_file) 
        
        self.training = True 
        self.epochs = 10

    def buildModel(self):
        """
        Builds a shallow neural network for Q-value prediction.
        """
        print("Building Agent Model...")

        # Step 1: Define the neural network
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))  # First hidden layer
        model.add(Dense(64, activation='relu'))               # Second hidden layer
        model.add(Dense(self.action_size, activation='linear'))              # Output layer

        # Step 2: Compile the model with Mean Squared Error loss and Adam optimizer
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        return model
    
    def experience(self, state, action, reward, next_state, done):
        """
        Stores experience in replay_buffer for replay.
        """
        self.replay_buffer.append((state, action, reward, next_state, done))

    def action(self, state, valid_moves):
        """
        Returns next action based on epsilon-greedy.
        """
        if not self.training:
            state = np.array(state).reshape(1, -1)
            q_values = self.model.predict(state, verbose=0)[0]
            for i in range(len(q_values)):
                if i not in valid_moves:
                    q_values[i] = float('-inf')
            return np.argmax(q_values)

        if random.random() <= self.epsilon:
            return random.choice(valid_moves)
        
        state = np.array(state).reshape(1, -1)
        q_values = self.model.predict(state, verbose=0)[0]
        for i in range(len(q_values)):
            if i not in valid_moves:
                q_values[i] = float('-inf')
        return np.argmax(q_values)

    def move(self, state):
        """
        Determines Player 2's move based on the current state of the game.

        Parameters:
        state (list): A list representing the current state of the TicTacToe board.

        Returns:
        int: The position (0-8) where Player 2 wants to make a move.
        """
        valid_moves = [i for i, val in enumerate(state) if val == 0]
        return self.action(state, valid_moves)
    
    def calculate_move_value(self, current_game, position):
        """
        Evaluates a potential move's strategic value using reward shaping.
        Higher rewards are given for more advantageous positions.
        """
        simulation = TicTacToe(smartMovePlayer1, playerSQN=self)
        simulation.board = current_game.board.copy()
        
        simulation.board[position] = 1 
        if simulation.check_winner(1):
            return 0.3
        
        simulation.board[position] = 2 
        if simulation.check_winner(2):
            return 0.5
        
        if position == 4 and current_game.board[4] == 0: 
            return 0.2
        
        if position in [0, 2, 6, 8] and current_game.board[position] == 0:
            return 0.1
        
        return 0.0

    def train(self, num_episodes=3000):
        """
        Trains the SQN through by first collecting experiences and then training on the stored experiences.
        """
        for episode in range(num_episodes):
            game = TicTacToe(smartMovePlayer1, playerSQN=self)
            state = game.board.copy()
            done = False
            
            game.player1_move()
            next_state = game.board.copy()
            
            while not done:
                valid_moves = game.empty_positions()
                action = self.move(state)
                
                if action in valid_moves:
                    immediate_reward = self.calculate_move_value(game, action)
                    
                    game.make_move(action, 2)
                    
                    if game.is_full() or game.current_winner is not None:
                        final_reward = game.get_reward() + immediate_reward
                        done = True
                        self.experience(state, action, final_reward, game.board.copy(), done)
                    else:
                        game.player1_move()
                        next_state = game.board.copy()
                        
                        if game.is_full() or game.current_winner is not None:
                            final_reward = game.get_reward() + immediate_reward
                            done = True
                        else:
                            final_reward = immediate_reward
                        
                        self.experience(state, action, final_reward, next_state, done)
                
                state = next_state.copy() if not done else next_state
            
            if episode % 10 == 0:
                print(f"Episode: {episode}, Epsilon: {self.epsilon:.2f}")

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            else:
                self.epsilon = self.epsilon_min

        print("Training on collected experiences...")
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            self.batch_training(32)

    def batch_training(self, batch_size):
        """
        Trains the model using experience replay.
        """
        if len(self.replay_buffer) < batch_size:
            return
        
        minibatch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = np.array(next_state).reshape(1, -1)
                target = reward + self.GAMMA * np.amax(self.model.predict(next_state, verbose=0)[0])
            
            state = np.array(state).reshape(1, -1)
            target_function = self.model.predict(state, verbose=0)
            target_function[0][action] = target
            self.model.fit(state, target_function, epochs=1, verbose=0)

    def save_model(self, filepath='2022A7PS1263G_MODEL.h5'):
        """
        Saves the weights of the trained model to a file.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "2022A7PS1263G_MODEL.h5")
        print("Model saved.")
        self.model.save(filepath)

    def load_model(self, filepath='2022A7PS1263G_MODEL.h5'):
     """
     Loads the weights of trained model from a file or creates a new model if loading fails.
     """
     try:
         current_dir = os.path.dirname(os.path.abspath(__file__))
         model_path = os.path.join(current_dir, "2022A7PS1263G_MODEL.h5")
         model = tf.keras.models.load_model(filepath, compile=False)
         model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
         print(f"Model successfully loaded from {filepath}")
         return model
     except Exception as e:
        print(f"Error loading model: {e}")
        print("Starting with a new model.")
        return self.buildModel()  

    
def main(smartMovePlayer1):
    """
    Simulates a TicTacToe game between Player 1 (random move player) and Player 2 (SQN-based player).

    Parameters:
    smartMovePlayer1: Probability that Player 1 will make a smart move at each time step.
                     During a smart move, Player 1 either tries to win the game or block the opponent.
    """
    num = 25
    for idx in range(1, num+1):
        playerSQN = PlayerSQN()
        
        print("Training the SQN agent...")
        playerSQN.train(num_episodes=2000)
        
        playerSQN.save_model('2022A7PS1263G_MODEL.h5')
        
        print("\nEvaluating the trained model...")
        playerSQN.training = False
        playerSQN.epsilon = 0 
        
        total_rewards = 0
        wins = 0
        losses = 0
        draws = 0
        num_games = 20
        for game_num in range(1, num_games + 1):
            print(f"Game {game_num}/{num_games}")
            game = TicTacToe(smartMovePlayer1, playerSQN)
            game.play_game()
            reward = game.get_reward()
            total_rewards += reward
            
            if game.current_winner == 2:
                wins += 1
            elif game.current_winner == 1:
                losses += 1
            else:
                draws += 1
            
            print(f"Reward for SQN Agent (Player 2) in Game {game_num}: {reward}")
        
        print(f"\nTotal Reward for SQN Agent (Player 2) over {num_games} games: {total_rewards}")
        print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")

        

if __name__ == "__main__":
    try:
        smartMovePlayer1 = float(sys.argv[1])
        assert 0 <= smartMovePlayer1 <= 1
    except:
        print("Usage: python 2022A7PS1263G.py <smartMovePlayer1Probability>")
        print("Example: python 2022A7PS1263G.py 0.5")
        print("There is an error. Probability must lie between 0 and 1.")
        sys.exit(1)
    
    main(smartMovePlayer1)