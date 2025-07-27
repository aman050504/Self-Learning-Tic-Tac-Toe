import os
import threading
from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# --- 1. Initialize Flask App ---
app = Flask(__name__)

# --- 2. Load Your AI Model ---
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2022A7PS1263G_MODEL.h5")

# --- NEW: Add a lock for thread-safe model saving and training ---
model_lock = threading.Lock()

try:
    # Load the model without compiling it initially.
    model = load_model(model_path, compile=False)
    
    # Manually re-compile the model after loading.
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

    print("-----------------------------------------")
    print("Keras model loaded and compiled successfully!")
    print("-----------------------------------------")
except Exception as e:
    print(f"Error loading Keras model: {e}")
    model = None

# --- 3. Define Training Constants ---
GAMMA = 0.95 # Discount factor for future rewards

# --- 4. Define Application Routes ---

@app.route('/')
def home():
    """
    This function handles requests to the root URL ('/').
    It serves the main HTML page for the game.
    """
    return render_template('tictactoe.html')

@app.route('/get_ai_move', methods=['POST'])
def get_ai_move():
    """
    This function handles the AI move requests from the frontend.
    """
    if not model:
        return jsonify({'error': 'Model is not loaded or failed to load.'}), 500

    data = request.get_json()
    if not data or 'board' not in data:
        return jsonify({'error': 'Invalid request: board data missing.'}), 400

    board_state = np.array(data['board']).reshape(1, -1)
    q_values = model.predict(board_state, verbose=0)[0]

    valid_moves = [i for i, v in enumerate(data['board']) if v == 0]
    
    if not valid_moves:
        return jsonify({'move': None, 'status': 'No valid moves'})

    for i in range(len(q_values)):
        if i not in valid_moves:
            q_values[i] = -np.inf

    ai_move = int(np.argmax(q_values))
    return jsonify({'move': ai_move})

# --- NEW: Add a route for continuous training ---
@app.route('/train_model', methods=['POST'])
def train_model():
    """
    Receives the final game experience from the frontend and uses it
    to perform a single step of training on the model.
    """
    if not model:
        return jsonify({'error': 'Model is not loaded.'}), 500

    data = request.get_json()
    
    # Extract experience components from the request
    state = np.array(data['state']).reshape(1, -1)
    action = data['action']
    reward = data['reward']
    next_state = np.array(data['next_state']).reshape(1, -1)
    done = data['done']

    # --- Q-Learning Bellman Equation ---
    # Predict the Q-values for the current state.
    target_f = model.predict(state, verbose=0)

    # Calculate the target Q-value for the action taken.
    # If the game is 'done', the future reward is 0.
    if done:
        target = reward
    else:
        # This part is for completeness, but our frontend will always send done=True
        q_next = model.predict(next_state, verbose=0)[0]
        target = reward + GAMMA * np.amax(q_next)
    
    # Update the Q-value for the specific action that was taken.
    target_f[0][action] = target

    # --- Train and Save the Model ---
    # Use a lock to ensure that model training and saving are atomic operations.
    with model_lock:
        # Train the model on this single experience.
        model.fit(state, target_f, epochs=1, verbose=0)
        
        # Save the newly trained model back to the file.
        try:
            model.save(model_path)
            print("Model updated and saved with new experience.")
        except Exception as e:
            print(f"Error saving model: {e}")

    return jsonify({'status': 'training step completed'})


# --- 5. Run the Flask App ---
if __name__ == '__main__':
    # Using port 5001 to avoid potential "Address already in use" errors.
    app.run(host='0.0.0.0', port=5001, debug=True)
