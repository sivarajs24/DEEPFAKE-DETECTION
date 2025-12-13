"""
Training Progress Monitor
Monitor ongoing training progress
"""

import os
from pathlib import Path
import time


def monitor_training():
    """Monitor training progress from logs"""
    
    print("=" * 70)
    print("📊 Training Progress Monitor")
    print("=" * 70)
    
    log_dir = Path("logs")
    
    if not log_dir.exists():
        print("\n⚠️  No logs directory found. Training hasn't started yet.")
        return
    
    # Find latest checkpoint
    checkpoints = list(log_dir.glob("**/*.ckpt"))
    
    if not checkpoints:
        print("\n⏳ Training in progress... No checkpoints yet.")
        print("💡 Checkpoints are saved after each epoch completes.")
        return
    
    latest_ckpt = max(checkpoints, key=os.path.getmtime)
    
    print(f"\n📁 Latest checkpoint: {latest_ckpt.name}")
    print(f"📅 Last modified: {time.ctime(os.path.getmtime(latest_ckpt))}")
    
    # Check tensorboard logs
    tb_logs = list(log_dir.glob("**/events.out.tfevents.*"))
    
    if tb_logs:
        print(f"\n✅ TensorBoard logs found: {len(tb_logs)} files")
        print("\n💡 View training graphs:")
        print("   tensorboard --logdir logs/")
        print("   Then open: http://localhost:6006")
    
    # Check wandb logs
    wandb_dir = Path("wandb")
    if wandb_dir.exists():
        print(f"\n✅ WandB logs found")
        print("💡 View at: https://wandb.ai/")
    
    print("\n" + "=" * 70)
    print("🔄 Training Status")
    print("=" * 70)
    
    # Estimate progress
    all_checkpoints = sorted(checkpoints, key=os.path.getmtime)
    
    if len(all_checkpoints) > 1:
        print(f"\n✅ Epochs completed: {len(all_checkpoints)}")
        
        # Calculate time per epoch
        first_time = os.path.getmtime(all_checkpoints[0])
        last_time = os.path.getmtime(all_checkpoints[-1])
        time_diff = last_time - first_time
        avg_time_per_epoch = time_diff / (len(all_checkpoints) - 1) if len(all_checkpoints) > 1 else 0
        
        if avg_time_per_epoch > 0:
            print(f"⏱️  Average time per epoch: {avg_time_per_epoch / 60:.1f} minutes")
            
            # Estimate remaining time (assuming 50 epochs)
            total_epochs = 50
            remaining_epochs = total_epochs - len(all_checkpoints)
            estimated_time = remaining_epochs * avg_time_per_epoch
            
            print(f"⏳ Estimated time remaining: {estimated_time / 3600:.1f} hours")
    
    print("\n💡 To check GPU usage (if using GPU):")
    print("   nvidia-smi")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    monitor_training()
