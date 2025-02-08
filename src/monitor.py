#!/usr/bin/env python
"""
Training monitoring dashboard.
"""

import os
import sys
import time
import json
import psutil
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, List
import subprocess
import GPUtil
from pathlib import Path

class TrainingMonitor:
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.log_dir = "logs"
        self.model_dir = f"outputs/granite-3.1-2b-{dataset}"
        self.start_time = datetime.now()
        
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU utilization and memory stats."""
        gpus = GPUtil.getGPUs()
        return {
            f"gpu_{i}": {
                "utilization": gpu.load * 100,
                "memory_used": gpu.memoryUsed,
                "memory_total": gpu.memoryTotal,
                "temperature": gpu.temperature
            }
            for i, gpu in enumerate(gpus)
        }
    
    def get_latest_checkpoint(self) -> Dict[str, Any]:
        """Find the latest checkpoint and metadata."""
        try:
            checkpoints = sorted(
                [d for d in os.listdir(self.model_dir) 
                 if d.startswith("checkpoint-")],
                key=lambda x: int(x.split("-")[1])
            )
            if not checkpoints:
                return {"status": "No checkpoints found"}
            
            latest = checkpoints[-1]
            checkpoint_path = Path(self.model_dir) / latest
            
            return {
                "checkpoint": latest,
                "step": int(latest.split("-")[1]),
                "size": sum(f.stat().st_size for f in checkpoint_path.rglob("*") if f.is_file()) / (1024 * 1024 * 1024),  # Size in GB
                "time": datetime.fromtimestamp(checkpoint_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            return {"status": f"Error getting checkpoint info: {str(e)}"}
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics from logs."""
        try:
            latest_log = sorted(
                [f for f in os.listdir(self.log_dir) 
                 if f.startswith(f"train_logs_{self.dataset}")],
                key=lambda f: os.path.getmtime(os.path.join(self.log_dir, f))
            )[-1]
            
            with open(os.path.join(self.log_dir, latest_log)) as f:
                lines = f.readlines()
                
            # Parse rewards and other metrics
            rewards: List[Dict] = []
            errors: List[str] = []
            current_step = 0
            
            for line in lines:
                if "Rewards:" in line:
                    try:
                        rewards.append(json.loads(line.split("Rewards:")[-1]))
                    except:
                        continue
                elif "step" in line.lower():
                    try:
                        step = int(line.split("step")[-1].split()[0])
                        if step > current_step:
                            current_step = step
                    except:
                        continue
                elif "error" in line.lower() or "exception" in line.lower():
                    errors.append(line.strip())
            
            return {
                "current_step": current_step,
                "latest_rewards": rewards[-1] if rewards else None,
                "avg_rewards": {
                    k: sum(r[k] for r in rewards[-100:]) / len(rewards[-100:])
                    for k in rewards[-1].keys()
                } if rewards else None,
                "recent_errors": errors[-5:] if errors else [],
                "log_file": latest_log
            }
        except Exception as e:
            return {"status": f"Error getting training stats: {str(e)}"}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system resource usage stats."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "running_time": str(datetime.now() - self.start_time)
        }
    
    def get_tmux_status(self) -> Dict[str, Any]:
        """Get status of tmux training session."""
        try:
            result = subprocess.run(
                ["tmux", "list-sessions"], 
                capture_output=True, 
                text=True
            )
            sessions = result.stdout.split('\n')
            training_session = None
            for session in sessions:
                if f"granite_training_{self.dataset}" in session:
                    training_session = session
                    break
            
            return {
                "active": bool(training_session),
                "session": training_session
            }
        except Exception as e:
            return {"status": f"Error getting tmux status: {str(e)}"}
    
    def display_status(self) -> None:
        """Display current training status."""
        os.system('clear')
        print(f"=== Training Monitor for {self.dataset} ===")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n=== System Status ===")
        system_stats = self.get_system_stats()
        print(f"CPU Usage: {system_stats['cpu_percent']}%")
        print(f"Memory Usage: {system_stats['memory_percent']}%")
        print(f"Disk Usage: {system_stats['disk_percent']}%")
        print(f"Running Time: {system_stats['running_time']}")
        
        print("\n=== GPU Status ===")
        gpu_stats = self.get_gpu_stats()
        for gpu, stats in gpu_stats.items():
            print(f"{gpu}:")
            print(f"  Utilization: {stats['utilization']:.1f}%")
            print(f"  Memory: {stats['memory_used']}/{stats['memory_total']} MB")
            print(f"  Temperature: {stats['temperature']}°C")
        
        print("\n=== Training Status ===")
        checkpoint_info = self.get_latest_checkpoint()
        print("Latest Checkpoint:")
        for k, v in checkpoint_info.items():
            print(f"  {k}: {v}")
        
        training_stats = self.get_training_stats()
        print("\nTraining Progress:")
        for k, v in training_stats.items():
            if k != "recent_errors":
                print(f"  {k}: {v}")
        
        if training_stats.get("recent_errors"):
            print("\nRecent Errors:")
            for error in training_stats["recent_errors"]:
                print(f"  {error}")
        
        tmux_status = self.get_tmux_status()
        print("\nTmux Status:")
        for k, v in tmux_status.items():
            print(f"  {k}: {v}")

def main():
    parser = argparse.ArgumentParser(description="Monitor GRPO training progress")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                      choices=["gsm8k", "ragbench"],
                      help="Dataset to monitor")
    parser.add_argument("--refresh", type=int, default=30,
                      help="Refresh interval in seconds")
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.dataset)
    
    try:
        while True:
            monitor.display_status()
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    main()