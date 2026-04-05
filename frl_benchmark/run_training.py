"""Training runner for FRL Benchmark.

Calls `flower-simulation` via subprocess, passing config from environment
variables set by the dashboard. Matches the original local-mode invocation
used in the dashboard but runs inside the Docker training container.
"""

import os
import shutil
import subprocess
import sys
import tempfile


def main() -> None:
    # ── Read config from environment variables ────────────────────────────────
    env_name = os.environ.get("FRL_ENV", "CartPole-v1")
    method = os.environ.get("FRL_METHOD", "fedpg-br")
    num_workers = int(os.environ.get("FRL_WORKERS", "10"))
    num_byzantine = int(os.environ.get("FRL_BYZANTINE", "0"))
    num_rounds = int(os.environ.get("FRL_ROUNDS", "50"))
    attack_type = os.environ.get("FRL_ATTACK", "random-noise")
    round_timeout = int(os.environ.get("FRL_ROUND_TIMEOUT", "600"))
    seed = int(os.environ.get("FRL_SEED", "42"))

    # Optional advanced overrides
    batch_size = int(os.environ.get("FRL_BATCH_SIZE", "0"))
    lr = float(os.environ.get("FRL_LR", "0"))
    sigma = float(os.environ.get("FRL_SIGMA", "0"))
    gamma = float(os.environ.get("FRL_GAMMA", "0"))
    mini_batch_size = int(os.environ.get("FRL_MINI_BATCH_SIZE", "0"))
    delta = float(os.environ.get("FRL_DELTA", "0"))
    max_episode_len = int(os.environ.get("FRL_MAX_EPISODE_LEN", "0"))

    # Build run-config string
    run_config_parts = [
        f'env="{env_name}"',
        f'method="{method}"',
        f"num-server-rounds={num_rounds}",
        f"num-workers={num_workers}",
        f"num-byzantine={num_byzantine}",
        f'attack-type="{attack_type}"',
        f"seed={seed}",
        f"round-timeout={round_timeout}",
    ]
    if batch_size > 0:
        run_config_parts.append(f"batch-size={batch_size}")
    if lr > 0:
        run_config_parts.append(f"lr={lr}")
    if sigma > 0:
        run_config_parts.append(f"sigma={sigma}")
    if gamma > 0:
        run_config_parts.append(f"gamma={gamma}")
    if mini_batch_size > 0:
        run_config_parts.append(f"mini-batch-size={mini_batch_size}")
    if delta > 0:
        run_config_parts.append(f"delta={delta}")
    if max_episode_len > 0:
        run_config_parts.append(f"max-episode-len={max_episode_len}")

    run_config_str = " ".join(run_config_parts)

    print(f"[FRL] Starting: env={env_name}, method={method}, "
          f"workers={num_workers}, byzantine={num_byzantine}, rounds={num_rounds}",
          flush=True)

    flower_sim = shutil.which("flower-simulation")
    if not flower_sim:
        print("ERROR: flower-simulation not found. Is flwr[simulation] installed?",
              flush=True)
        sys.exit(1)

    cmd = [
        flower_sim,
        "--app", "/app",
        "--num-supernodes", str(num_workers),
        "--run-config", run_config_str,
    ]

    # Fresh HOME dir per run — prevents SQLite state conflicts between experiments
    tmp_home = tempfile.mkdtemp(prefix="flwr-run-")
    try:
        env = os.environ.copy()
        env["HOME"] = tmp_home
        result = subprocess.run(cmd, env=env)
        sys.exit(result.returncode)
    finally:
        shutil.rmtree(tmp_home, ignore_errors=True)


if __name__ == "__main__":
    main()
