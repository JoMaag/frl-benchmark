"""Web dashboard for Flower FRL Benchmark.

Real-time visualization of federated learning with multiple clients.
"""

import logging
import os
import re
import subprocess
import sys
import threading
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("werkzeug").setLevel(logging.ERROR)
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, jsonify, redirect, render_template, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config["SECRET_KEY"] = "frl-benchmark-secret"
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
active_clients: Dict[str, dict] = {}
training_metrics: List[dict] = []
current_round = 0
_experiment_process: Optional[subprocess.Popen] = None
_experiment_thread: Optional[threading.Thread] = None


@app.route("/")
def index():
    """Redirect to experiment dashboard."""
    return redirect("/experiment")


@app.route("/experiment")
def experiment():
    """Serve experiment configuration dashboard."""
    return render_template("experiment.html")


@app.route("/api/strategies")
def get_strategies():
    """List all available aggregation strategies."""
    try:
        from frl_benchmark.strategies import list_strategies
        return jsonify(list_strategies())
    except ImportError:
        # Fallback if strategies module not loaded
        return jsonify({
            "fedpg-br": "Byzantine filtering + SCSG variance reduction",
            "svrpg": "SCSG variance reduction, no Byzantine filtering",
            "gpomdp": "Simple averaging, single gradient step (baseline)",
        })


@app.route("/api/status")
def get_status():
    """Get current training status."""
    return jsonify(
        {
            "active_clients": len(active_clients),
            "current_round": current_round,
            "total_rounds": 50,
            "status": "running" if active_clients else "idle",
        }
    )


@app.route("/api/clients")
def get_clients():
    """Get list of active clients."""
    return jsonify(list(active_clients.values()))


@app.route("/api/metrics")
def get_metrics():
    """Get training metrics history."""
    return jsonify(training_metrics)


@app.route("/api/metrics", methods=["POST"])
def push_metrics():
    """Receive metrics pushed from the training server."""
    global current_round
    data = request.get_json(silent=True) or {}
    round_num = data.get("round", 0)
    if round_num > current_round:
        current_round = round_num
    metric_entry = {"timestamp": datetime.now().isoformat(), **data}
    training_metrics.append(metric_entry)
    if len(training_metrics) > 100:
        training_metrics.pop(0)
    socketio.emit("metrics", metric_entry)
    if data.get("done"):
        socketio.emit("experiment_done", {
            "method": data.get("method", ""),
            "exit_code": 0,
            "final_reward": data.get("server_avg_reward", 0),
        })
    return jsonify({"ok": True})


@socketio.on("connect")
def handle_connect():
    """Handle client connection."""
    emit("status", {"message": "Connected to Flower FRL Benchmark dashboard"})


def _docker_restart_training(
    env_name: str,
    method: str,
    num_rounds: int,
    num_workers: int,
    num_byzantine: int,
    attack_type: Optional[str] = None,
    batch_size=None,
    learning_rate=None,
    sigma=None,
    gamma=None,
    mini_batch_size=None,
    delta=None,
    max_episode_len=None,
    hidden_units=None,
    activation=None,
    _seed=42,
) -> bool:
    """Stop the training container and restart flower-simulation with new config."""
    global training_metrics, current_round
    try:
        import socket as _socket
        import docker
        try:
            client = docker.from_env()
        except Exception as docker_err:
            socketio.emit("log", {"message": "ERROR: Docker is not running. Please start Docker Desktop and try again."})
            return False
        project = os.environ.get("COMPOSE_PROJECT_NAME", "frl_benchmark")

        # 1. Stop existing training container (if any)
        import time as _time
        try:
            old = client.containers.get("frl-benchmark-server")
            socketio.emit("log", {"message": "Stopping existing training..."})
            old.stop(timeout=10)
            old.remove(force=True)
            # Wait for Docker to finish the removal before creating a new container
            # with the same name (avoids 409 Conflict race condition)
            for _ in range(40):
                try:
                    client.containers.get("frl-benchmark-server")
                    _time.sleep(0.1)
                except docker.errors.NotFound:
                    break
        except docker.errors.NotFound:
            pass  # No previous container — nothing to stop

        # 2. Clear metrics for fresh experiment
        training_metrics.clear()
        current_round = 0

        # 3. Get image + network + runs volume from our own container (same image)
        my_container = client.containers.get(_socket.gethostname())
        image_id = my_container.attrs["Image"]
        network_name = next(iter(my_container.attrs["NetworkSettings"]["Networks"]))
        network_obj = client.networks.get(network_name)

        # Find the frl-runs volume mounted in this container so training can share it
        runs_volume = next(
            (m["Name"] for m in my_container.attrs.get("Mounts", [])
             if m.get("Destination") == "/app/runs" and m.get("Type") == "volume"),
            None,
        )

        # 4. Build env vars for run_training.py (bypasses flwr run / SuperLink / SQLite)
        attack_cfg = attack_type if attack_type and attack_type != "none" else "random-noise"
        train_env = {
            "PYTHONUNBUFFERED": "1",
            "DASHBOARD_URL": "http://dashboard:8050",
            "FRL_ENV": env_name,
            "FRL_METHOD": method,
            "FRL_WORKERS": str(num_workers),
            "FRL_BYZANTINE": str(num_byzantine),
            "FRL_ROUNDS": str(num_rounds),
            "FRL_ATTACK": attack_cfg,
            "FRL_SEED": str(_seed),
        }
        if batch_size:
            train_env["FRL_BATCH_SIZE"] = str(int(batch_size))
        if learning_rate:
            train_env["FRL_LR"] = str(float(learning_rate))
        if sigma:
            train_env["FRL_SIGMA"] = str(float(sigma))
        if gamma:
            train_env["FRL_GAMMA"] = str(float(gamma))
        if mini_batch_size:
            train_env["FRL_MINI_BATCH_SIZE"] = str(int(mini_batch_size))
        if delta:
            train_env["FRL_DELTA"] = str(float(delta))
        if max_episode_len:
            train_env["FRL_MAX_EPISODE_LEN"] = str(int(max_episode_len))

        socketio.emit("log", {"message": (
            f"Starting: env={env_name}, method={method}, "
            f"rounds={num_rounds}, workers={num_workers}, byzantine={num_byzantine}"
        )})
        volumes = {}
        if runs_volume:
            volumes[runs_volume] = {"bind": "/app/runs", "mode": "rw"}

        container = client.containers.create(
            image=image_id,
            command=["python", "/app/frl_benchmark/run_training.py"],
            environment=train_env,
            name="frl-benchmark-server",
            shm_size="2g",
            volumes=volumes,
            labels={
                "com.docker.compose.project": project,
                "com.docker.compose.service": "server",
            },
        )
        network_obj.connect(container)
        container.start()
        socketio.emit("log", {"message": f"Training started with {num_workers} simulated workers."})

        # Stream training container logs back to the browser
        def _stream_docker_logs():
            round_loss_re = re.compile(r"\bround (\d+): (-?[\d.]+)$")
            round_detail_re = re.compile(
                r"Round (\d+)(?: \[(\w[\w-]*)\])?: good_agents=(\d+), scsg_steps=(\d+), active=(\d+), skipped=(\d+)"
            )
            last_good_agents = 0
            last_scsg_steps = 0
            last_reward = 0.0
            in_history = False
            try:
                for raw in container.logs(stream=True, follow=True):
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    if not line:
                        continue
                    clean = re.sub(r'\x1b\[[0-9;]*m', '', line)

                    # Suppress Flower's end-of-training history dump
                    if "History (" in clean:
                        in_history = True
                    if in_history:
                        continue

                    m = round_detail_re.search(clean)
                    if m:
                        last_good_agents = int(m.group(3))
                        last_scsg_steps = int(m.group(4))
                        socketio.emit("log", {"message": clean})
                        continue

                    m = round_loss_re.search(clean)
                    if m:
                        round_num = int(m.group(1))
                        reward = -float(m.group(2))  # Flower logs loss = -reward
                        last_reward = reward
                        display = re.sub(r"\bround \d+: -?[\d.]+$",
                                         f"round {round_num}: {reward:.1f}", clean)
                        socketio.emit("log", {"message": display})
                        continue
            except Exception as log_err:
                socketio.emit("log", {"message": f"[log stream ended: {log_err}]"})

            # Container finished
            try:
                result = container.wait()
                exit_code = result.get("StatusCode", -1)
            except Exception:
                exit_code = -1
            socketio.emit("experiment_done", {
                "method": method,
                "exit_code": exit_code,
                "final_reward": last_reward,
            })
            socketio.emit("log", {"message": f"Experiment finished (exit code {exit_code})"})

        import threading as _threading
        _threading.Thread(target=_stream_docker_logs, daemon=True).start()
        return True

    except ImportError:
        socketio.emit("log", {"message": "ERROR: docker package missing — rebuild image."})
        return False
    except Exception as e:
        socketio.emit("log", {"message": f"ERROR restarting training: {e}"})
        return False


@socketio.on("start_experiment")
def handle_start_experiment(data):
    """Launch a flower-simulation subprocess from the dashboard."""
    global _experiment_process, _experiment_thread

    # Kill any running experiment first
    _kill_experiment()

    env_name = data.get("env", "CartPole-v1")
    method = data.get("method", "fedpg-br")
    num_workers = int(data.get("num_workers", 10))
    num_byzantine = int(data.get("num_byzantine", 0))
    num_rounds = int(data.get("num_rounds", 312))
    attack_type = data.get("attack_type", "none")
    batch_size = data.get("batch_size")
    learning_rate = data.get("learning_rate")
    sigma = data.get("sigma")
    gamma = data.get("gamma")
    mini_batch_size = data.get("mini_batch_size")
    delta = data.get("delta")
    max_episode_len = data.get("max_episode_len")
    hidden_units = data.get("hidden_units")
    activation = data.get("activation")
    round_timeout = int(data.get("round_timeout", 600))
    seed = int(data.get("seed", 42))

    # ── Docker-compose mode ──────────────────────────────────────────────────
    # Stop existing server/workers and restart with the dashboard's parameters.
    if os.environ.get("RUNNING_IN_DOCKER"):
        _docker_restart_training(
            env_name=env_name, method=method, num_rounds=num_rounds,
            num_workers=num_workers, num_byzantine=num_byzantine,
            attack_type=attack_type, batch_size=batch_size,
            learning_rate=learning_rate, sigma=sigma, gamma=gamma,
            mini_batch_size=mini_batch_size, delta=delta,
            max_episode_len=max_episode_len, hidden_units=hidden_units,
            activation=activation,
            _seed=seed,
        )
        return
    # ─────────────────────────────────────────────────────────────────────────

    use_frl_benchmark = "true" if method == "fedpg-br" else "false"
    attack_cfg = attack_type if attack_type != "none" else "random-noise"

    # Flower 1.27+ format: strings use double quotes, bools/ints bare
    run_config = (
        f'env="{env_name}" method="{method}" '
        f"num-server-rounds={num_rounds} num-workers={num_workers} "
        f'num-byzantine={num_byzantine} use-fedpg-br={use_frl_benchmark} '
        f'attack-type="{attack_cfg}"'
    )
    if batch_size:
        run_config += f" batch-size={int(batch_size)}"
    if learning_rate:
        run_config += f" lr={float(learning_rate)}"
    if sigma:
        run_config += f" sigma={float(sigma)}"
    if gamma:
        run_config += f" gamma={float(gamma)}"
    if mini_batch_size:
        run_config += f" mini-batch-size={int(mini_batch_size)}"
    if delta:
        run_config += f" delta={float(delta)}"
    if max_episode_len:
        run_config += f" max-episode-len={int(max_episode_len)}"
    if hidden_units:
        run_config += f' hidden-units="{hidden_units}"'
    if activation:
        run_config += f' activation="{activation}"'
    run_config += f" round-timeout={round_timeout}"

    # Find the project root (where pyproject.toml is)
    project_root = Path(__file__).resolve().parent.parent.parent

    # Find flower-simulation executable next to the running Python interpreter
    # On conda/venv Windows: python.exe is in env root, scripts are in Scripts/
    python_dir = Path(sys.executable).parent
    candidates = [
        python_dir / "Scripts" / "flwr-simulation.exe",    # Windows conda/venv (new)
        python_dir / "Scripts" / "flwr-simulation",
        python_dir / "Scripts" / "flower-simulation.exe",  # Windows conda/venv (old)
        python_dir / "Scripts" / "flower-simulation",
        python_dir / "flwr-simulation.exe",                # Some installations
        python_dir / "flwr-simulation",                    # Linux/macOS (new)
        python_dir / "flower-simulation.exe",
        python_dir / "flower-simulation",                  # Linux/macOS (old)
    ]
    flower_sim = None
    for c in candidates:
        if c.exists():
            flower_sim = str(c)
            break

    if not flower_sim:
        import shutil
        flower_sim = shutil.which("flwr-simulation") or shutil.which("flower-simulation")

    # Use new `flwr run` API if old simulation binary not found
    if not flower_sim:
        import shutil
        flwr_bin = shutil.which("flwr")
        if flwr_bin:
            cmd = [flwr_bin, "run", str(project_root), "-c", run_config]
        else:
            socketio.emit("log", {"message": f"ERROR: Cannot find flwr or flower-simulation. Python at: {sys.executable}"})
            return
    else:
        cmd = [flower_sim, "--app", str(project_root), "--run-config", run_config]

    print(f"[Dashboard] Launching: {cmd}")
    socketio.emit("log", {"message": f"CMD: {cmd[0]}"})
    socketio.emit("log", {"message": f"Launching {method.upper()} on {env_name} (K={num_workers}, B={num_byzantine}, rounds={num_rounds})"})

    try:
        _experiment_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(project_root),
        )
    except Exception as e:
        socketio.emit("log", {"message": f"ERROR: Failed to launch: {e}"})
        return

    # Stream output in a background thread
    def stream_output():
        global _experiment_process
        proc = _experiment_process
        if proc is None or proc.stdout is None:
            return

        # Regex patterns for parsing flower output
        round_loss_re = re.compile(r"\bround (\d+): (-?[\d.]+)$")
        round_detail_re = re.compile(
            r"Round (\d+)(?: \[(\w[\w-]*)\])?: good_agents=(\d+), scsg_steps=(\d+), active=(\d+), skipped=(\d+)"
        )

        last_good_agents = 0
        last_scsg_steps = 0
        last_reward = 0.0
        in_history = False

        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue

            # Strip ANSI color codes for parsing
            clean = re.sub(r'\x1b\[[0-9;]*m', '', line)

            # Suppress Flower's end-of-training history dump
            if "History (" in clean:
                in_history = True
            if in_history:
                continue

            # Parse round detail line
            m = round_detail_re.search(clean)
            if m:
                last_good_agents = int(m.group(3))
                last_scsg_steps = int(m.group(4))
                socketio.emit("log", {"message": clean})
                continue

            # Parse Flower 1.20 fit_progress: "round N: loss" (loss = -reward)
            m = round_loss_re.search(clean)
            if m:
                round_num = int(m.group(1))
                reward = -float(m.group(2))
                last_reward = reward
                display = re.sub(r"\bround \d+: -?[\d.]+$",
                                 f"round {round_num}: {reward:.1f}", clean)
                socketio.emit("log", {"message": display})
                continue

        # Process finished
        exit_code = proc.wait()
        socketio.emit("experiment_done", {
            "method": method,
            "exit_code": exit_code,
            "final_reward": last_reward,
        })
        socketio.emit("log", {"message": f"Experiment finished (exit code {exit_code})"})
        _experiment_process = None

    _experiment_thread = threading.Thread(target=stream_output, daemon=True)
    _experiment_thread.start()


@socketio.on("stop_experiment")
def handle_stop_experiment():
    """Stop the running experiment."""
    if os.environ.get("RUNNING_IN_DOCKER"):
        try:
            import docker
            client = docker.from_env()
            try:
                ctr = client.containers.get("frl-benchmark-server")
                ctr.stop(timeout=5)
                try:
                    ctr.remove(force=True)
                except docker.errors.APIError:
                    pass  # Already being removed (409 conflict) — ignore
                socketio.emit("log", {"message": "Experiment stopped by user."})
            except docker.errors.NotFound:
                socketio.emit("log", {"message": "No running experiment to stop."})
        except Exception as e:
            socketio.emit("log", {"message": f"Error stopping experiment: {e}"})
    else:
        _kill_experiment()
        socketio.emit("log", {"message": "Experiment stopped by user."})


def _kill_experiment():
    """Terminate the running experiment process if any."""
    global _experiment_process
    if _experiment_process is not None:
        try:
            _experiment_process.terminate()
            _experiment_process.wait(timeout=5)
        except Exception:
            try:
                _experiment_process.kill()
            except Exception:
                pass
        _experiment_process = None


@socketio.on("register_client")
def handle_register_client(data):
    """Register a new training client."""
    client_id = data.get("client_id")
    client_info = {
        "id": client_id,
        "name": data.get("name", f"Client {client_id}"),
        "location": data.get("location", "Unknown"),
        "status": "active",
        "last_seen": datetime.now().isoformat(),
    }
    active_clients[client_id] = client_info

    # Broadcast update to all connected dashboards
    socketio.emit("client_update", {"clients": list(active_clients.values())})


@socketio.on("metrics_update")
def handle_metrics_update(data):
    """Handle real-time metrics update from training."""
    global current_round

    round_num = data.get("round")
    metrics = data.get("metrics", {})

    if round_num > current_round:
        current_round = round_num

    # Store metrics
    metric_entry = {
        "round": round_num,
        "timestamp": datetime.now().isoformat(),
        **metrics,
    }
    training_metrics.append(metric_entry)

    # Keep only last 100 entries
    if len(training_metrics) > 100:
        training_metrics.pop(0)

    # Broadcast to all connected dashboards
    socketio.emit("metrics", metric_entry)


def _start_tensorboard():
    """Launch TensorBoard on port 6006 watching the runs/ directory."""
    import subprocess as _sp
    runs_dir = "/app/runs" if os.environ.get("RUNNING_IN_DOCKER") else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "runs"))
    os.makedirs(runs_dir, exist_ok=True)
    print(f"[TensorBoard] logdir={runs_dir}", flush=True)
    _sp.Popen(
        ["tensorboard", "--logdir", runs_dir, "--host", "0.0.0.0", "--port", "6006"],
        stdout=_sp.DEVNULL,
        stderr=_sp.DEVNULL,
    )


def start_dashboard(host: str = "0.0.0.0", port: int = 8050):

    import webbrowser
    import threading

    _start_tensorboard()

    url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}/experiment"
    print(f"\n Flower FRL Benchmark Dashboard starting at {url}")

    # Open browser after a short delay to let the server start
    if not os.environ.get("RUNNING_IN_DOCKER"):
        threading.Timer(1.5, webbrowser.open, args=[url]).start()

    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    start_dashboard()
