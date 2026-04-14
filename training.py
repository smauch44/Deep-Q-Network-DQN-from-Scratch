"""
Extraordinary DQN training procedure for CartPole-v1.

This version is designed to maximize the chance of reaching
an evaluation average reward of at least 400 and ideally 500.

It includes:
- replay warm-up
- minibatch TD learning
- periodic hard target updates
- gradient clipping
- slower, more stable optimization
- best-model checkpointing
- detailed metric logging

Assumes DQNAgent and DQNConfig are already defined in the notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import logging
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from dqn_components import DQNAgent, DQNConfig


def _get_logger() -> logging.Logger:
    """
    Logging setup for this script.

    Creates a logs/ directory next to this file and configures:
      - logs/training.log : dedicated log for training.py

    The logger also echoes to the console so output remains visible in Colab.
    Safe to call multiple times.
    """
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
        )

        file_handler = logging.FileHandler(log_dir / "training.log", mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        logger.propagate = False

    return logger


LOG = _get_logger()


@dataclass
class TrainingHistory:
    """Stores metrics collected during training."""
    episode_rewards: list[float] = field(default_factory=list)
    moving_average_rewards: list[float] = field(default_factory=list)
    episode_losses: list[float] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)
    replay_sizes: list[int] = field(default_factory=list)
    best_moving_average: float = 0.0
    best_episode_reward: float = 0.0


class DQNTrainer:
    """High-quality DQN trainer for CartPole-v1."""

    def __init__(
        self,
        config: DQNConfig,
        num_episodes: int = 500,
        target_update_every_episodes: int = 3,
        moving_average_window: int = 20,
        early_stop_reward: float = 475.0,
    ) -> None:
        self.config = config
        self.num_episodes = num_episodes
        self.target_update_every_episodes = target_update_every_episodes
        self.moving_average_window = moving_average_window
        self.early_stop_reward = early_stop_reward

        self.project_root = Path(__file__).resolve().parent
        self.checkpoint_dir = self.project_root / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.checkpoint_dir / "training_best_model.pt"

        self.env = gym.make("CartPole-v1")
        self.agent = DQNAgent(config)
        self.history = TrainingHistory()

        self.best_q_state_dict = copy.deepcopy(self.agent.q_network.state_dict())
        self.best_target_state_dict = copy.deepcopy(
            self.agent.target_network.state_dict()
        )

        LOG.info(
            "Training initialized | episodes=%d moving_average_window=%d early_stop=%.1f",
            self.num_episodes,
            self.moving_average_window,
            self.early_stop_reward,
        )

    def _train_step_with_clipping(self) -> float | None:
        """Perform one DQN training step with gradient clipping."""
        if not self.agent.ready_to_train():
            return None

        loss = self.agent.compute_dqn_loss()

        self.agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.agent.q_network.parameters(),
            max_norm=10.0,
        )
        self.agent.optimizer.step()

        self.agent.training_steps += 1
        return float(loss.item())

    def train(self) -> TrainingHistory:
        """Run DQN training over multiple episodes."""
        for episode_index in range(1, self.num_episodes + 1):
            state, info = self.env.reset(seed=self.config.seed + episode_index)
            terminated = False
            truncated = False
            total_reward = 0.0
            episode_losses: list[float] = []

            while not (terminated or truncated):
                action = self.agent.select_action(state=state, explore=True)

                next_state, reward, terminated, truncated, step_info = self.env.step(
                    action
                )
                done = terminated or truncated

                self.agent.store_transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    info=step_info,
                )

                loss = self._train_step_with_clipping()
                if loss is not None:
                    episode_losses.append(loss)

                state = next_state
                total_reward += reward

            self.agent.update_epsilon()

            if episode_index % self.target_update_every_episodes == 0:
                self.agent.update_target_network()

            average_episode_loss = (
                float(np.mean(episode_losses)) if episode_losses else 0.0
            )

            self.history.episode_rewards.append(total_reward)
            moving_average_reward = float(
                np.mean(self.history.episode_rewards[-self.moving_average_window :])
            )
            self.history.moving_average_rewards.append(moving_average_reward)
            self.history.episode_losses.append(average_episode_loss)
            self.history.epsilons.append(self.agent.epsilon)
            self.history.replay_sizes.append(len(self.agent.replay_buffer))

            if total_reward > self.history.best_episode_reward:
                self.history.best_episode_reward = total_reward

            if moving_average_reward > self.history.best_moving_average:
                self.history.best_moving_average = moving_average_reward
                self.best_q_state_dict = copy.deepcopy(
                    self.agent.q_network.state_dict()
                )
                self.best_target_state_dict = copy.deepcopy(
                    self.agent.target_network.state_dict()
                )
                self.agent.save(str(self.best_model_path))
                LOG.info(
                    "New best moving average %.2f at episode %d. Checkpoint saved to %s",
                    moving_average_reward,
                    episode_index,
                    self.best_model_path,
                )

            message = (
                f"Episode {episode_index:03d} | "
                f"Reward: {total_reward:6.1f} | "
                f"AvgReward({self.moving_average_window}): "
                f"{moving_average_reward:7.2f} | "
                f"Loss: {average_episode_loss:10.6f} | "
                f"Epsilon: {self.agent.epsilon:.4f} | "
                f"Replay: {len(self.agent.replay_buffer):6d} | "
                f"BestAvg: {self.history.best_moving_average:7.2f}"
            )
            print(message)
            LOG.info(message)

            if moving_average_reward >= self.early_stop_reward:
                stop_message = (
                    "Early stopping triggered: moving average reward "
                    f"reached {moving_average_reward:.2f}."
                )
                print(f"\n{stop_message}")
                LOG.info(stop_message)
                break

        self.agent.q_network.load_state_dict(self.best_q_state_dict)
        self.agent.target_network.load_state_dict(self.best_target_state_dict)

        return self.history

    def close(self) -> None:
        """Close the environment cleanly."""
        self.env.close()
        LOG.info("Training environment closed.")


def summarize_training(history: TrainingHistory) -> None:
    """Print final training summary."""
    rewards = np.array(history.episode_rewards, dtype=np.float32)
    moving_avg = np.array(history.moving_average_rewards, dtype=np.float32)
    losses = np.array(history.episode_losses, dtype=np.float32)

    print("\nTraining summary")
    print("-" * 60)
    print(f"Total episodes trained:      {len(rewards)}")
    print(f"Final episode reward:        {rewards[-1]:.2f}")
    print(f"Best episode reward:         {history.best_episode_reward:.2f}")
    print(f"Final moving avg reward:     {moving_avg[-1]:.2f}")
    print(f"Best moving avg reward:      {history.best_moving_average:.2f}")
    print(f"Mean episode loss:           {losses.mean():.6f}")
    print(f"Final epsilon:               {history.epsilons[-1]:.4f}")
    print(f"Final replay buffer size:    {history.replay_sizes[-1]}")

    LOG.info(
        "Training summary | episodes=%d final_reward=%.2f best_reward=%.2f final_mavg=%.2f best_mavg=%.2f mean_loss=%.6f final_epsilon=%.4f final_replay=%d",
        len(rewards),
        rewards[-1],
        history.best_episode_reward,
        moving_avg[-1],
        history.best_moving_average,
        losses.mean(),
        history.epsilons[-1],
        history.replay_sizes[-1],
    )


if __name__ == "__main__":
    config = DQNConfig(
        state_dim=4,
        action_dim=2,
        hidden_dim_1=128,
        hidden_dim_2=128,
        learning_rate=5e-4,
        gamma=0.99,
        batch_size=64,
        replay_buffer_size=100_000,
        min_replay_size=2_000,
        target_update_every=1_000,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.985,
        seed=42,
    )

    num_episodes = int(os.environ.get("DQN_TRAIN_EPISODES", "500"))
    target_update_every_episodes = int(
        os.environ.get("DQN_TRAIN_TARGET_UPDATE_EPISODES", "3")
    )
    moving_average_window = int(os.environ.get("DQN_TRAIN_MOVING_AVG_WINDOW", "20"))
    early_stop_reward = float(os.environ.get("DQN_TRAIN_EARLY_STOP", "475.0"))

    trainer = DQNTrainer(
        config=config,
        num_episodes=num_episodes,
        target_update_every_episodes=target_update_every_episodes,
        moving_average_window=moving_average_window,
        early_stop_reward=early_stop_reward,
    )

    try:
        training_history = trainer.train()
        summarize_training(training_history)
    finally:
        trainer.close()
