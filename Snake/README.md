# Snake game with AI model
#### Author: Bruce Smith


INSERT GIF


Welcome to Snake with a Deep Q-Learning model that is designed to beat the game. This project combines the classic Snake game experience with the power of artificial intelligence. The game itself offers an engaging and nostalgic experience of controlling a snake, while the AI model aims to learn and achieve high scores.

## Features

- Play the classic Snake game using arrow keys to control the snake's movement.
- A custom-trained AI model that learns from gameplay and attempts to maximize its score.
- Modular and extensible codebase, allowing you to experiment with different AI algorithms and strategies.
- Visualization of AI's decision-making process and learning progress.

## How to play

1. Clone this repository and install dependencies.
2. In game.py, set dqn_play flag to False to play yourself or True to watch the AI play.
3. Run game.py.
4. If playing yourself, use the arrow keys to control the snake. Else, watch the AI learn to play the game. 

## AI Model
The AI model is trained to play the Snake game using Reinforcement Learning, specifically a Deep Q-Network (DQN) algorithm. The model observes the game state, learns the best actions to take, and improves its performance over time.

* __dqnmodel.py__ holds the Deep Q-learning model iself along with the training steps.
* __snakedqn.py__ holds the Snake specific details of training the model. The model showcased uses an input size of 11 (game state), a single hidden layer of size 256, and an output size of 3 (action).
* __replaymemory.py__ holds the ReplayMemory object used to store past experiences to better train the AI

Hyperparameters used:
- Learning rate = 0.001
- Discount rate = 0.97
- Initial Epsilon = 1
- Epsilon linear decay rate: 0.0001

## Credits
These sources were used to aid in the devlopment of the AI
 - [Geeks for Geeks](https://www.geeksforgeeks.org/ai-driven-snake-game-using-deep-q-learning/#)
 - [Deep reinforcement learning paper by Anton Finnson and Victor Molno](https://www.diva-portal.org/smash/get/diva2:1342302/FULLTEXT01.pdf)
