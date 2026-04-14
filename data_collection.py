"""
Part 3: Data Collection for DQN on CartPole-v1

This notebook cell demonstrates how to:
- create the CartPole-v1 environment
- select actions using epsilon-greedy exploration
- step through the environment with env.step(action)
- store transitions in the replay buffer
- loop over multiple episodes
- track total reward per episode

Assumes DQNAgent and DQNConfig are already defined in the notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path

import gymnasium as gym
import numpy as np

from dqn_components import DQNAgent, DQNConfig


def _get_logger() -> logging.Logger:
    """
    Logging setup for this script.

    Creates a logs/ directory next to this file and configures:
      - logs/data_collection.log : dedicated log for data_collection.py

    The logger also echoes to the console so output remains visible in Colab.
    Safe to call multiple times.
    """
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("data_collection")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
        )

        file_handler = logging.FileHandler(
            log_dir / "data_collection.log",
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
class EpisodeResult:
    """Summary statistics for one completed episode."""

    episode_index: int
    total_reward: float
    episode_length: int
    terminated: bool
    truncated: bool


class DQNTrainer:
    """
    Handles environment interaction and data collection for DQN.

    Responsibilities:
    - create and manage the CartPole-v1 environment
    - run full episodes
    - use the agent to select actions
    - step through the environment
    - store transitions in replay memory
    - track total reward and episode length
    """

    def __init__(self, config: DQNConfig) -> None:
        self.config = config
        self.env = gym.make("CartPole-v1")
        self.agent = DQNAgent(config)

    def run_single_episode(self, episode_index: int) -> EpisodeResult:
        """Run one full episode and collect transitions."""
        state, info = self.env.reset(seed=self.config.seed + episode_index)
        total_reward = 0.0
        episode_length = 0
        terminated = False
        truncated = False

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

            state = next_state
            total_reward += reward
            episode_length += 1

        self.agent.update_epsilon()

        result = EpisodeResult(
            episode_index=episode_index,
            total_reward=total_reward,
            episode_length=episode_length,
            terminated=terminated,
            truncated=truncated,
        )
        LOG.info(
            "Episode %03d | reward=%.1f length=%d epsilon=%.4f replay_size=%d",
            result.episode_index,
            result.total_reward,
            result.episode_length,
            self.agent.epsilon,
            len(self.agent.replay_buffer),
        )
        return result

    def collect_experience(self, num_episodes: int) -> list[EpisodeResult]:
        """Run multiple episodes and collect transitions in replay memory."""
        results: list[EpisodeResult] = []

        for episode_index in range(1, num_episodes + 1):
            result = self.run_single_episode(episode_index=episode_index)
            results.append(result)

            print(
                f"Episode {result.episode_index:03d} | "
                f"Reward: {result.total_reward:6.1f} | "
                f"Length: {result.episode_length:3d} | "
                f"Epsilon: {self.agent.epsilon:.4f} | "
                f"Replay Size: {len(self.agent.replay_buffer):6d}"
            )

        return results

    def close(self) -> None:
        """Close the environment cleanly."""
        self.env.close()
        LOG.info("Data collection environment closed.")


def summarize_results(results: list[EpisodeResult]) -> None:
    """Print summary statistics for the collected episodes."""
    if not results:
        print("No episodes were collected.")
        LOG.warning("No episodes were collected.")
        return

    rewards = np.array(
        [result.total_reward for result in results],
        dtype=np.float32,
    )
    lengths = np.array(
        [result.episode_length for result in results],
        dtype=np.int32,
    )

    print("\nData collection summary")
    print("-" * 40)
    print(f"Episodes collected: {len(results)}")
    print(f"Average reward:     {rewards.mean():.2f}")
    print(f"Max reward:         {rewards.max():.2f}")
    print(f"Min reward:         {rewards.min():.2f}")
    print(f"Average length:     {lengths.mean():.2f}")
    print(f"Max length:         {lengths.max()}")
    print(f"Min length:         {lengths.min()}")

    LOG.info(
        "Summary | episodes=%d avg_reward=%.2f max_reward=%.2f min_reward=%.2f avg_length=%.2f max_length=%d min_length=%d",
        len(results),
        rewards.mean(),
        rewards.max(),
        rewards.min(),
        lengths.mean(),
        lengths.max(),
        lengths.min(),
    )


if __name__ == "__main__":
    config = DQNConfig()
    trainer = DQNTrainer(config)

    num_episodes = int(os.environ.get("DQN_DATA_COLLECTION_EPISODES", "10"))
    LOG.info("Starting data collection for %d episodes.", num_episodes)

    try:
        results = trainer.collect_experience(num_episodes=num_episodes)
        summarize_results(results)

        print("\nReplay buffer verification")
        print("-" * 40)
        print(f"Transitions stored: {len(trainer.agent.replay_buffer)}")
        print(f"Ready to train:     {trainer.agent.ready_to_train()}")

        LOG.info(
            "Replay buffer verification | transitions=%d ready_to_train=%s",
            len(trainer.agent.replay_buffer),
            trainer.agent.ready_to_train(),
        )
    finally:
        trainer.close()
