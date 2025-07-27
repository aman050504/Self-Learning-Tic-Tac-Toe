# Noughts & Neurons: A Self-Learning Tic-Tac-Toe AI

This project is a full-stack web application that allows users to play Tic-Tac-Toe against an AI powered by a deep reinforcement learning model. The AI continuously learns and improves from the games it plays against users.

## Features

* **Intelligent AI Opponent**: The AI uses a Deep Q-Network (DQN) built with Keras/TensorFlow to make strategic moves.
* **Interactive Web Interface**: A clean, modern, and responsive UI built with HTML, CSS, and JavaScript allows for smooth gameplay directly in the browser.
* **Full-Stack Architecture**: The application uses a Python Flask server to host the AI model and handle game logic, cleanly separating the frontend from the backend.
* **RESTful API**: The frontend communicates with the backend via a simple REST API to get the AI's moves.
* **Sleek UI/UX**: The interface features a premium dark theme with subtle animations for player moves and a dynamic line to indicate a win.

## Technologies Used

* **Backend**: Python, Flask, TensorFlow, Keras, NumPy
* **Frontend**: HTML, CSS (Tailwind CSS), JavaScript
* **AI Model**: Deep Q-Network (DQN)

---

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### 1. Prerequisites

Make sure you have Python (3.8 or newer) and `pip` installed on your system.

### 2. Clone the Repository

First, clone this repository to your local machine using Git:

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 3. Set Up a Virtual Environment (Recommended)

It's highly recommended to create a virtual environment to keep the project's dependencies isolated.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies

Install all the required Python libraries using the provided `requirements.txt` file (or install them manually).

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` file, you can install the packages manually:

```bash
pip install Flask tensorflow numpy
```

### 5. Run the Application

Once the dependencies are installed, you can start the Flask web server:

```bash
python app.py
```

You should see output in your terminal indicating that the server is running, usually on port 5001:

```
* Serving Flask app 'app'
* Running on [http://127.0.0.1:5001](http://127.0.0.1:5001)
(Press CTRL+C to quit)
```

### 6. Play the Game!

Open your web browser and navigate to the following address:

[http://127.0.0.1:5001](http://127.0.0.1:5001)

You can now play against the AI!

---

## How It Works: Code Structure

The project is organized into a frontend and a backend, which communicate with each other.

### Backend (`app.py`)

* **Flask Application**: `app.py` is the core of the backend. It's a Python script that creates a Flask web server.
* **Model Loading**: When the server starts, it loads the pre-trained Keras model (`2022A7PS1263G_MODEL.h5`) into memory.
* **API Endpoints**:
    * `@app.route('/')`: This route serves the main `tictactoe.html` page to the user's browser.
    * `@app.route('/get_ai_move', methods=['POST'])`: This is the API endpoint for the AI. The frontend sends the current board state here. The backend uses the loaded model to predict the best move and sends it back to the frontend as a JSON response.

### Frontend (`templates/tictactoe.html`)

* **Structure (HTML)**: Defines the layout of the game, including the title, the 3x3 grid for the board, the status display, and the restart button.
* **Styling (CSS)**: Uses the Tailwind CSS framework for a modern, responsive design. Custom styles are included for the 'X' and 'O' symbols, hover effects, and the winning strike-through line animation.
* **Interactivity (JavaScript)**:
    * **Game Logic**: Manages the game state, player turns, and checks for win/draw conditions.
    * **DOM Manipulation**: Updates the visual board when a player or the AI makes a move.
    * **API Communication**: Uses the `fetch()` function to send a `POST` request to the `/get_ai_move` endpoint on the backend. It sends the current board as JSON data and waits for the AI's move in response.

### AI Model (`2022A7PS1263G.py`, `TicTacToe.py`, `2022A7PS1263G_MODEL.h5`)

* **`TicTacToe.py`**: Contains the fundamental rules and logic of the Tic-Tac-Toe game.
* **`2022A7PS1263G.py`**: Defines the `PlayerSQN` class, which is the agent that uses the deep learning model. This script contains the logic for training the model using experience replay.
* **`2022A7PS1263G_MODEL.h5`**: This is the pre-trained, saved Keras model file containing the "brain" of the AI. It's what the Flask server loads to make predictions.
