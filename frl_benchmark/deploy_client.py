"""Distributed deployment client for Flower FRL Benchmark.

This script runs a client in deployment mode, connecting to a remote server
(e.g., a separate Docker container or machine).
"""

import argparse
from logging import INFO

import flwr as fl
from flwr.common.logger import log

from frl_benchmark.config import get_config
from frl_benchmark.flower.worker import Worker
from frl_benchmark.client_app import FRLClient


def start_client(
    server_address: str,
    client_id: int,
    env_name: str = "CartPole-v1",
    is_byzantine: bool = False,
    attack_type: str = "random-noise",
) -> None:
    """Start Flower client in deployment mode.

    Args:
        server_address: Server address (host:port)
        client_id: Unique client ID
        env_name: Environment name
        is_byzantine: Whether this is a Byzantine client
        attack_type: Type of attack if Byzantine
    """
    log(INFO, f"Starting client {client_id}")
    log(INFO, f"Connecting to server: {server_address}")
    log(INFO, f"Environment: {env_name}")
    if is_byzantine:
        log(INFO, f"Byzantine mode: {attack_type}")

    config = get_config(env_name)

    worker = Worker(
        worker_id=client_id,
        env_name=env_name,
        hidden_units=config.hidden_units,
        gamma=config.gamma,
        activation=config.activation,
        output_activation=config.output_activation,
        is_byzantine=is_byzantine,
        attack_type=attack_type if is_byzantine else None,
        max_episode_len=config.max_episode_len,
        device="cpu",
    )

    client = FRLClient(worker=worker)

    fl.client.start_client(
        server_address=server_address,
        client=client.to_client(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower FRL Benchmark Deployment Client")
    parser.add_argument("--server", type=str, required=True, help="Server address (host:port)")
    parser.add_argument("--id", type=int, required=True, help="Client ID")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--byzantine", action="store_true", help="Run as Byzantine client")
    parser.add_argument(
        "--attack", type=str, default="random-noise",
        choices=["random-noise", "sign-flip", "random-action", "fedpg-attack",
                 "variance-attack", "zero-gradient", "reward-flipping"],
    )

    args = parser.parse_args()

    start_client(
        server_address=args.server,
        client_id=args.id,
        env_name=args.env,
        is_byzantine=args.byzantine,
        attack_type=args.attack,
    )
