"""Distributed deployment server for Flower FRL Benchmark.

This script runs the server in deployment mode, listening for real client connections
across the network (Docker containers, separate machines, etc.).
"""

import argparse
from logging import INFO

import flwr as fl
from flwr.common.logger import log
from flwr.server import ServerConfig

from frl_benchmark.server_app import FRLStrategy


def start_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = 50,
    num_clients: int = 10,
    env_name: str = "CartPole-v1",
    method: str = "fedpg-br",
    byzantine_ratio: float = 0.0,
) -> None:
    """Start Flower server in deployment mode.

    Args:
        server_address: Address to listen on (host:port)
        num_rounds: Number of training rounds
        num_clients: Expected number of clients
        env_name: Environment name
        method: Aggregation method (fedpg-br, svrpg, gpomdp)
        byzantine_ratio: Ratio of Byzantine clients
    """
    log(INFO, f"Starting Flower FRL Benchmark server on {server_address}")
    log(INFO, f"Environment: {env_name}")
    log(INFO, f"Method: {method}")
    log(INFO, f"Expected clients: {num_clients}")
    log(INFO, f"Training rounds: {num_rounds}")
    log(INFO, f"Byzantine ratio: {byzantine_ratio}")

    strategy = FRLStrategy(
        env_name=env_name,
        num_agents=num_clients,
        byzantine_ratio=byzantine_ratio,
        use_adaptive_batch=(method == "fedpg-br"),
        method=method,
    )

    config = ServerConfig(num_rounds=num_rounds)

    fl.server.start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower FRL Benchmark Deployment Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--rounds", type=int, default=312)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument(
        "--method", type=str, default="fedpg-br",
        choices=["fedpg-br", "svrpg", "gpomdp"],
        help="Aggregation strategy",
    )
    parser.add_argument("--byzantine-ratio", type=float, default=0.0)

    args = parser.parse_args()

    start_server(
        server_address=f"{args.host}:{args.port}",
        num_rounds=args.rounds,
        num_clients=args.clients,
        env_name=args.env,
        method=args.method,
        byzantine_ratio=args.byzantine_ratio,
    )
