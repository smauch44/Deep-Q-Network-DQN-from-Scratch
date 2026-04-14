# Deep Q-Network (DQN) from Scratch — CartPole-v1

## Overview
This project implements a **Deep Q-Network (DQN)** from scratch using **PyTorch** and applies it to the classic **CartPole-v1** environment from Gymnasium.

The objective is to train an agent to balance a pole on a moving cart by learning an optimal policy through interaction with the environment. The implementation follows the original DQN framework with key components such as:
- Experience Replay
- Target Network
- Epsilon-Greedy Exploration
- Temporal Difference Learning

The final trained agent achieves **maximum performance (reward = 500)** and consistently exceeds the required evaluation threshold.

---

## Environment Setup

### Requirements
- Python 3.10+
- PyTorch
- Gymnasium

Install dependencies using:
```bash
pip install -r requirements.txt
```

### Environment Initialization
```python
import gymnasium as gym
env = gym.make("CartPole-v1")
```

- State space: 4-dimensional continuous vector  
- Action space: 2 discrete actions (left, right)

---

## Project Structure

```
dqn/
├── checkpoints/
├── logs/
├── data_collection.py
├── dqn_components.py
├── env_setup.py
├── evaluation.py
├── training.py
├── requirements.txt
└── README.md
```

---

## DQN Architecture

- Input: 4  
- Hidden layers: 128, 128 (ReLU)  
- Output: 2  

---

## Core Components

### Q-Network & Target Network
- Online network for learning  
- Target network for stable targets  
- Hard update every 1000 steps  

### Replay Buffer
Stores transitions:
(state, action, reward, next_state, done, info)

### Epsilon-Greedy Policy
- Start: 1.0  
- End: 0.05  
- Decay: 0.995  

---

## Training

DQN loss:
(r + γ max Q_target(s', a') - Q_online(s, a))^2

- Optimizer: Adam  
- Batch size: 64  
- Gamma: 0.99  

---

## Results

- Final reward: 500  
- Avg reward: ~490  
- Evaluation reward: 500  

---

## Run Instructions

```bash
python env_setup.py
python data_collection.py
python training.py
python evaluation.py
```

---

## Conclusion

This implementation successfully trains a DQN agent from scratch and achieves optimal performance on CartPole-v1.
