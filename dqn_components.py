"""
Deep Q-Network components for CartPole-v1.

This module implements:
- DQNConfig: central hyperparameter configuration
- QNetwork: multilayer perceptron for Q-value approximation
- ReplayBuffer: experience replay memory
- DQNAgent: online network, target network, action selection, and updates

Designed for:
- Python 3.10+
- PyTorch
- CartPole-v1 (state_dim=4, action_dim=2)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch import optim


def _get_logger() -> logging.Logger:
    """
    Logging setup for this script.

    Creates a logs/ directory next to this file and configures:
      - logs/dqn_components.log : dedicated log for dqn_components.py

    The logger also echoes to the console so output remains visible in Colab.
    Safe to call multiple times.
    """
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("dqn_components")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
        )

        file_handler = logging.FileHandler(
            log_dir / "dqn_components.log",
            mode="a",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        logger.propagate = False

    return logger


LOG = _get_logger()


@dataclass
class DQNConfig:
    """
    Configuration for DQN hyperparameters.

    These values are chosen to be strong defaults for CartPole-v1 while
    remaining faithful to standard DQN design principles from the original
    paper: replay memory, target network, bootstrapped TD learning, and
    epsilon-greedy exploration.
    """

    state_dim: int = 4
    action_dim: int = 2

    hidden_dim_1: int = 128
    hidden_dim_2: int = 128

    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64

    replay_buffer_size: int = 100_000
    min_replay_size: int = 1_000

    target_update_every: int = 1_000

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Transition:
    """A single environment transition stored in replay memory."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    info: dict[str, Any]


class QNetwork(nn.Module):
    """
    Small multilayer perceptron used to approximate Q(s, a).

    Architecture:
        input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> output

    For CartPole-v1:
        input dimension = 4
        output dimension = 2
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 128,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-values for all discrete actions."""
        return self.network(state)


class ReplayBuffer:
    """
    Experience replay buffer for DQN.

    Stores transitions of the form:
        (state, action, reward, next_state, done, info)

    Sampling random mini-batches breaks temporal correlation between
    consecutive experiences and improves training stability.
    """

    def __init__(self, capacity: int) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Add one transition to replay memory."""
        transition = Transition(
            state=np.asarray(state, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_state=np.asarray(next_state, dtype=np.float32),
            done=bool(done),
            info=info if info is not None else {},
        )
        self.buffer.append(transition)

    def sample(
        self,
        batch_size: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        """Sample a random mini-batch of transitions."""
        batch = random.sample(self.buffer, batch_size)

        states = np.array([transition.state for transition in batch], dtype=np.float32)
        actions = np.array([transition.action for transition in batch], dtype=np.int64)
        rewards = np.array([transition.reward for transition in batch], dtype=np.float32)
        next_states = np.array(
            [transition.next_state for transition in batch],
            dtype=np.float32,
        )
        dones = np.array([transition.done for transition in batch], dtype=np.float32)
        infos = [transition.info for transition in batch]

        return states, actions, rewards, next_states, dones, infos

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent for CartPole-v1.

    This class maintains:
    - an online Q-network for action selection and learning
    - a target Q-network for stable TD targets
    - a replay buffer for experience replay
    - an optimizer for gradient-based updates

    Notes:
    - Target network updates are periodic hard updates.
    - Action selection uses epsilon-greedy exploration.
    """

    def __init__(self, config: DQNConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

        self._set_random_seeds(config.seed)

        self.q_network = QNetwork(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim_1=config.hidden_dim_1,
            hidden_dim_2=config.hidden_dim_2,
        ).to(self.device)

        self.target_network = QNetwork(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim_1=config.hidden_dim_1,
            hidden_dim_2=config.hidden_dim_2,
        ).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate,
        )
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)

        self.epsilon = config.epsilon_start
        self.training_steps = 0

        LOG.info(
            "DQNAgent initialized | device=%s epsilon_start=%.4f buffer_size=%s",
            self.device,
            self.epsilon,
            config.replay_buffer_size,
        )

    @staticmethod
    def _set_random_seeds(seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """Select an action using epsilon-greedy exploration."""
        if explore and random.random() < self.epsilon:
            return random.randrange(self.config.action_dim)

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())

        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Store one transition in the replay buffer."""
        self.replay_buffer.push(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            info=info,
        )

    def update_epsilon(self) -> None:
        """
        Decay epsilon after training steps or episodes.

        Epsilon is clipped so it never drops below epsilon_end.
        """
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay,
        )

    def ready_to_train(self) -> bool:
        """Check whether replay memory contains enough samples to begin training."""
        return len(self.replay_buffer) >= self.config.min_replay_size

    def sample_batch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a mini-batch from replay memory and convert to tensors."""
        states, actions, rewards, next_states, dones, _ = self.replay_buffer.sample(
            self.config.batch_size
        )

        states_tensor = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=self.device,
        )
        actions_tensor = torch.as_tensor(
            actions,
            dtype=torch.int64,
            device=self.device,
        ).unsqueeze(1)
        rewards_tensor = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        next_states_tensor = torch.as_tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )
        dones_tensor = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        return (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
        )

    def compute_dqn_loss(self) -> torch.Tensor:
        """Compute the standard DQN mean-squared TD loss."""
        states, actions, rewards, next_states, dones = self.sample_batch()

        current_q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(
                dim=1,
                keepdim=True,
            )[0]
            target_q_values = rewards + (
                self.config.gamma * max_next_q_values * (1.0 - dones)
            )

        loss = self.loss_fn(current_q_values, target_q_values)
        return loss

    def train_step(self) -> float | None:
        """Perform one DQN optimization step if enough replay data exists."""
        if not self.ready_to_train():
            return None

        loss = self.compute_dqn_loss()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_steps += 1

        if self.training_steps % self.config.target_update_every == 0:
            self.update_target_network()

        return float(loss.item())

    def update_target_network(self) -> None:
        """Perform a hard update of the target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, filepath: str) -> None:
        """Save the online Q-network parameters to disk."""
        torch.save(self.q_network.state_dict(), filepath)
        LOG.info("Saved Q-network checkpoint to %s", filepath)

    def load(self, filepath: str) -> None:
        """
        Load parameters into the online Q-network and synchronize the target
        network immediately afterward.
        """
        state_dict = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(state_dict)
        self.update_target_network()
        LOG.info("Loaded Q-network checkpoint from %s", filepath)


if __name__ == "__main__":
    config = DQNConfig()
    agent = DQNAgent(config)

    print("DQN agent initialized successfully.")
    print("Device:", agent.device)
    print("Initial epsilon:", agent.epsilon)
    print("Replay buffer size:", len(agent.replay_buffer))
    print("Q-network:")
    print(agent.q_network)

    LOG.info("DQN components smoke run completed successfully.")
