"""
Part 1 environment verification for Assignment 6.

Creates a CartPole-v1 environment, resets it, takes a few random steps,
and verifies rendering with rgb_array mode.
"""

from __future__ import annotations

import logging
from pathlib import Path

import gymnasium as gym


def _get_logger() -> logging.Logger:
    """
    Logging setup for this script.

    Creates a logs/ directory next to this file and configures:
      - logs/env_setup.log : dedicated log for env_setup.py

    The logger also echoes to the console so output remains visible in Colab.
    Safe to call multiple times.
    """
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("env_setup")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
        )

        file_handler = logging.FileHandler(log_dir / "env_setup.log", mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        logger.propagate = False

    return logger


LOG = _get_logger()


def main() -> None:
    """Create the CartPole environment and verify basic interaction."""
    LOG.info("Starting CartPole-v1 environment verification.")
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    observation, info = env.reset(seed=42)
    print("Initial observation:", observation)
    print("Initial info:", info)
    print("Observation shape:", observation.shape)
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    LOG.info("Initial observation: %s", observation)
    LOG.info("Initial info: %s", info)
    LOG.info("Observation shape: %s", observation.shape)
    LOG.info("Action space: %s", env.action_space)
    LOG.info("Observation space: %s", env.observation_space)

    for step in range(5):
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, step_info = env.step(
            action
        )
        frame = env.render()

        print(f"\nStep {step + 1}")
        print("Action:", action)
        print("Next observation:", next_observation)
        print("Reward:", reward)
        print("Terminated:", terminated)
        print("Truncated:", truncated)
        print("Info:", step_info)
        print(
            "Rendered frame shape:",
            None if frame is None else frame.shape,
        )

        LOG.info(
            "Step %s | action=%s reward=%s terminated=%s truncated=%s frame_shape=%s",
            step + 1,
            action,
            reward,
            terminated,
            truncated,
            None if frame is None else frame.shape,
        )

        if terminated or truncated:
            print("\nEpisode ended early. Resetting environment.")
            LOG.info("Episode ended early at step %s. Resetting environment.", step + 1)
            next_observation, step_info = env.reset()

    env.close()
    LOG.info("Environment setup verification completed successfully.")
    print("\nEnvironment setup verification completed successfully.")


if __name__ == "__main__":
    main()
