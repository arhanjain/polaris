"""
Batch evaluation script using Python dataclass configs.

Usage:
    python scripts/batch_eval.py --config experiments.my_experiment
    python scripts/batch_eval.py --config experiments.my_experiment --dry-run
    python scripts/batch_eval.py --config experiments.my_experiment --max-concurrent 2
"""

import subprocess
import sys
import os
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any
import tyro
import json

from polaris.config import EvalArgs, PolicyArgs, BatchConfig, PolicyServer, JobCfg

# Global tracking for cleanup on Ctrl-C
_active_processes: list[subprocess.Popen] = []
_interrupted = False


def _kill_process_tree(proc: subprocess.Popen):
    """Kill a process and all its children."""
    import signal as sig
    if proc.poll() is None:
        try:
            # Kill entire process group
            os.killpg(os.getpgid(proc.pid), sig.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), sig.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()


def _signal_handler(signum, frame):
    """Handle Ctrl-C by killing all active subprocesses (jobs and their servers)."""
    global _interrupted
    _interrupted = True
    print("\n\n⚠ Interrupted! Killing all running processes...")
    
    for proc in _active_processes:
        _kill_process_tree(proc)
    
    print("All processes terminated.")
    sys.exit(1)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


@dataclass
class BatchArgs:
    """Arguments for batch evaluation."""
    config: str                                # Python module path (e.g., experiments.my_experiment)
    run_folder: str | None = None              # Shared run folder (auto-generated if None)
    max_concurrent: int = 1                    # Max concurrent jobs (Isaac Sim is heavy, often 1 is best)
    dry_run: bool = False                      # Print commands without running
    gpu_ids: list[int] = field(default_factory=lambda: [0])  # GPU IDs to use for round-robin


def load_config(config_path: str) -> BatchConfig:
    """Load BatchConfig from a Python file."""
    import importlib.util
    
    # Ensure it's a file path
    if not config_path.endswith(".py"):
        config_path = config_path.replace(".", "/") + ".py"
    
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load module from file path
    spec = importlib.util.spec_from_file_location("config_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if not hasattr(module, "config"):
        raise ValueError(f"Config file {config_path} must define a 'config' variable of type BatchConfig")
    
    config = module.config
    if not isinstance(config, BatchConfig):
        raise TypeError(f"config must be BatchConfig, got {type(config)}")
    
    return config


def find_free_port() -> int:
    """Find a free port by binding to port 0 and letting the OS assign one."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def build_command(eval_args: EvalArgs, run_folder: str) -> list[str]:
    """Build command-line arguments for eval.py from EvalArgs."""
    cmd = [sys.executable, "scripts/eval.py"]

    # cmd.append(f"--usd={eval_args.usd}")
    cmd.append(f"--policy.name={eval_args.policy.name}")
    cmd.append(f"--policy.client={eval_args.policy.client}")
    cmd.append(f"--policy.host={eval_args.policy.host}")
    cmd.append(f"--policy.port={eval_args.policy.port}")
    
    if eval_args.policy.open_loop_horizon is not None:
        cmd.append(f"--policy.open-loop-horizon={eval_args.policy.open_loop_horizon}")
    
    if eval_args.headless:
        cmd.append("--headless")
    else:
        cmd.append("--no-headless")
    
    cmd.append(f"--environment={eval_args.environment}")
    
    if eval_args.initial_conditions_file is not None:
        cmd.append(f"--initial-conditions-file={eval_args.initial_conditions_file}")
    
    if eval_args.instruction is not None:
        cmd.append(f"--instruction={eval_args.instruction}")
    
    cmd.append(f"--run-folder={run_folder}")

    return cmd

def build_command_all_args(eval_args: EvalArgs, run_folder: str) -> list[str]:
    """Automatically convert all EvalArgs fields to CLI arguments for eval.py."""
    import sys
    import shlex

    def flatten(prefix, obj):
        """Recursively flatten fields of a dataclass or dict to CLI args."""
        args = []
        if hasattr(obj, "__dataclass_fields__"):
            items = obj.__dict__.items()
        elif isinstance(obj, dict):
            items = obj.items()
        else:
            raise ValueError("Object to flatten should be a dataclass or dict")
        for k, v in items:
            if v is None:
                continue
            key = f"{prefix}.{k}" if prefix else k
            if hasattr(v, "__dataclass_fields__") or isinstance(v, dict):
                args.extend(flatten(key, v))
            elif isinstance(v, bool):
                if key == "headless":
                    # Preserve original convention for headless switches
                    args.append("--headless" if v else "--no-headless")
                else:
                    args.append(f"--{key}={str(v).lower()}")
            elif isinstance(v, list):
                # Assume repeated argument: --key item1 --key item2
                for item in v:
                    args.append(f"--{key}={shlex.quote(str(item))}")
            else:
                args.append(f"--{key}={shlex.quote(str(v))}")
        return args

    cmd = [sys.executable, "scripts/eval.py"]
    # Add all eval_args except run_folder
    cmd += flatten("", eval_args)
    # Add or override run-folder at the end
    cmd.append(f"--run-folder={shlex.quote(str(run_folder))}")
    return cmd


def launch_job_server(job: JobCfg, job_output_dir: Path, job_index: int, gpu_id: int, timeout: float = 300) -> subprocess.Popen | None:
    """Launch the server for a job and wait for it to be ready. Returns the process or None on failure."""
    global _active_processes
    
    if job.server is None:
        return None
    
    server = job.server
    
    # Assign a free port
    port = find_free_port()
    job.eval_args.policy.port = port
    
    # Replace {port} placeholder in command
    command = server.command.replace("{port}", str(port))
    
    server_log = job_output_dir / "server.log"
    print(f"[Job {job_index}] Launching server '{server.name}' on GPU {gpu_id}, port {port}")
    print(f"[Job {job_index}] Server log: {server_log}")
    
    # Server uses same GPU as the job
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
    }
    
    log_handle = open(server_log, "w")
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=Path(__file__).parent.parent,
        env=env,
        start_new_session=True,  # Create new process group so we can kill all children
    )
    _active_processes.append(proc)

    return proc
    # Wait for ready message
    # print(f"[Job {job_index}] Waiting for server ready: '{server.ready_message}'")
    # start_time = datetime.now()
    
    # import time
    # while (datetime.now() - start_time).total_seconds() < timeout:
    #     if proc.poll() is not None:
    #         print(f"[Job {job_index}] ✗ Server exited unexpectedly (code {proc.returncode})")
    #         return None
        
    #     log_handle.flush()
    #     with open(server_log, "r") as f:
    #         if server.ready_message in f.read():
    #             print(f"[Job {job_index}] ✓ Server ready on port {port}")
    #             return proc
        
    #     time.sleep(1)
    
    # print(f"[Job {job_index}] ✗ Server timeout")
    # proc.terminate()
    # return None


def shutdown_job_server(proc: subprocess.Popen | None, job_index: int):
    """Shutdown a job's server and all its child processes."""
    global _active_processes
    
    if proc is None:
        return
    
    if proc.poll() is None:
        print(f"[Job {job_index}] Shutting down server...")
        _kill_process_tree(proc)
    
    if proc in _active_processes:
        _active_processes.remove(proc)


def run_eval_job(job: JobCfg, run_folder: str, job_index: int, gpu_id: int = 0) -> dict[str, Any]:
    """Run a single evaluation job with its own server (if specified)."""
    global _active_processes, _interrupted
    
    eval_args = job.eval_args
    
    if _interrupted:
        return {"job_index": job_index, "return_code": -1, "interrupted": True}
    
    # Create job output directory
    job_output_dir = Path(run_folder) / eval_args.environment / eval_args.policy.name
    # job_output_dir = Path(run_folder) / Path(eval_args.usd).stem / eval_args.policy.name
    job_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Job {job_index}] === {eval_args.policy.name} on {eval_args.environment} ===")
    
    # Launch server for this job (on same GPU) if specified
    server_proc = None
    if job.server is not None:
        server_proc = launch_job_server(job, job_output_dir, job_index, gpu_id)
        if server_proc is None:
            return {
                "job_index": job_index,
                "policy_name": eval_args.policy.name,
                "environment": eval_args.environment,
                "return_code": -1,
                "duration_seconds": 0,
                "gpu_id": gpu_id,
                "error": "Server failed to start",
            }
    
    # Build command (after server assigned port to eval_args.policy.port)
    cmd = build_command(eval_args, run_folder)
    log_file = job_output_dir / "job.log"
    
    print(f"[Job {job_index}] Running eval...")
    print(f"[Job {job_index}] Log: {log_file}")
    
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
    }
    
    start_time = datetime.now()
    try:
        with open(log_file, "w") as f:
            f.write(f"Job {job_index}: {eval_args.policy.name} on {eval_args.environment}\n")
            f.write(f"GPU: cuda:{gpu_id}\n")
            f.write(f"Policy port: {eval_args.policy.port}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Started: {start_time.isoformat()}\n")
            f.write("=" * 60 + "\n\n")
            f.flush()
            
            proc = subprocess.Popen(
                cmd,
                cwd=Path(__file__).parent.parent,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
            _active_processes.append(proc)
            
            return_code = proc.wait()
            
            if proc in _active_processes:
                _active_processes.remove(proc)
        
        end_time = datetime.now()
        
        with open(log_file, "a") as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Finished: {end_time.isoformat()}\n")
            f.write(f"Duration: {(end_time - start_time).total_seconds():.1f}s\n")
            f.write(f"Exit code: {return_code}\n")
    
    finally:
        # Always shutdown server when job is done
        shutdown_job_server(server_proc, job_index)
    
    return {
        "job_index": job_index,
        "policy_name": eval_args.policy.name,
        "environment": eval_args.environment,
        "return_code": return_code,
        "duration_seconds": (end_time - start_time).total_seconds(),
        "gpu_id": gpu_id,
        "log_file": str(log_file),
    }


def main(args: BatchArgs):
    # Load job configurations
    batch_config = load_config(args.config)
    jobs = batch_config.jobs
    print(f"Loaded {len(jobs)} job(s) from {args.config}")
    
    # Create shared run folder
    if args.run_folder is None:
        run_folder = f"runs/{datetime.now().strftime('%Y-%m-%d')}/{datetime.now().strftime('%I:%M:%S %p')}"
    else:
        run_folder = args.run_folder
    
    run_folder_path = Path(run_folder)
    run_folder_path.mkdir(parents=True, exist_ok=True)
    
    # Save batch configuration for reproducibility
    batch_meta = {
        "config_module": args.config,
        "run_folder": run_folder,
        "start_time": datetime.now().isoformat(),
        "num_jobs": len(jobs),
        "max_concurrent": args.max_concurrent,
        "gpu_ids": args.gpu_ids,
        "jobs": [asdict(job) for job in jobs],
    }
    
    with open(run_folder_path / "batch_config.json", "w") as f:
        json.dump(batch_meta, f, indent=2)
    
    print(f"\n{'#'*60}")
    print(f"# Batch Evaluation")
    print(f"# Run folder: {run_folder}")
    print(f"# Jobs: {len(jobs)}")
    print(f"# Max concurrent: {args.max_concurrent}")
    print(f"# GPUs: {args.gpu_ids}")
    print(f"{'#'*60}\n")
    
    # Dry run - just print commands
    if args.dry_run:
        print("=== DRY RUN ===\n")
        print("Each job launches its own server (if specified), runs eval, then shuts down server.\n")
        
        for i, job in enumerate(jobs):
            eval_args = job.eval_args
            gpu_id = args.gpu_ids[i % len(args.gpu_ids)]
            print(f"[Job {i}] {eval_args.policy.name} on {eval_args.environment}")
            print(f"  GPU: {gpu_id}")
            if job.server:
                print(f"  Server: {job.server.name}")
                print(f"    {job.server.command.replace('{port}', '<auto>')}")
            else:
                print(f"  Server: none")
            cmd = build_command(eval_args, run_folder)
            print(f"  Eval: {' '.join(cmd)}\n")
        return
    
    # Run jobs (each job manages its own server lifecycle)
    results = []
    
    if args.max_concurrent == 1:
        # Sequential execution
        for i, job in enumerate(jobs):
            if _interrupted:
                break
            gpu_id = args.gpu_ids[i % len(args.gpu_ids)]
            result = run_eval_job(job, run_folder, i, gpu_id)
            results.append(result)
            status = "✓" if result["return_code"] == 0 else "✗"
            print(f"[Job {i}] {status} Completed in {result['duration_seconds']:.1f}s\n")
    else:
        # Parallel execution with ThreadPoolExecutor (shares memory for process tracking)
        with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
            futures = {}
            for i, job in enumerate(jobs):
                gpu_id = args.gpu_ids[i % len(args.gpu_ids)]
                future = executor.submit(run_eval_job, job, run_folder, i, gpu_id)
                futures[future] = i
            
            for future in as_completed(futures):
                if _interrupted:
                    break
                result = future.result()
                results.append(result)
                status = "✓" if result["return_code"] == 0 else "✗"
                print(f"[Job {result['job_index']}] {status} Completed in {result['duration_seconds']:.1f}s\n")
    
    # Summary
    successful = sum(1 for r in results if r["return_code"] == 0)
    failed = len(results) - successful
    
    print(f"\n{'#'*60}")
    print(f"# Batch evaluation complete!")
    print(f"# Results: {successful} succeeded, {failed} failed")
    print(f"# Saved to: {run_folder}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    args = tyro.cli(BatchArgs)
    main(args)
