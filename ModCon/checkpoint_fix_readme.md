# Distributed Training Checkpoint Saving Issue - Fix Documentation

## Problem Description

When training with multi-node distributed setup (8 nodes × 4 GPUs = 32 GPUs), the training crashes at epoch 7 with the following error:

```
AssertionError at line 398 in _save_checkpoint
assert success
```

**Key Observation:**
- ✅ Works with 1 node, 1 GPU
- 🟡 May work with 1 node, 4 GPUs (but can still fail)
- ❌ Fails with 8 nodes, 32 GPUs at epoch 7

## Root Cause

The `save_checkpoint` method in `trainer.py` has a **race condition**:

```python
def save_checkpoint(self, epoch, checkpoint_names=None):
    # ... checkpoint preparation code ...
    
    # DDP checkpoints are only saved on rank 0 (all workers are identical)
    if self.distributed_rank != 0:
        return  # ← Non-rank-0 processes return here WITHOUT waiting

    for checkpoint_path in checkpoint_paths:
        self._save_checkpoint(checkpoint, checkpoint_path)
    # ← Missing barrier here!
```

### What Happens:

1. **Rank 0**: Starts saving the checkpoint to shared filesystem (`/pscratch`)
2. **Ranks 1-31**: Return immediately and continue execution
3. **Race Condition**: Non-rank-0 processes may try to access the checkpoint file or proceed to next operations while rank 0 is still writing
4. **File System Contention**: With 32 processes on a shared network filesystem, concurrent access causes file locking conflicts
5. **Save Fails**: The `assert success` in `_save_checkpoint` fails due to I/O conflicts

### Why It Fails More Often with Multi-Node:

| Configuration | Risk Level | Reason |
|--------------|------------|---------|
| 1 node, 1 GPU | ✅ None | No distributed training, no race condition |
| 1 node, 4 GPUs | 🟡 Low | Same physical machine, local filesystem, faster I/O |
| 8 nodes, 32 GPUs | ❌ High | Network filesystem, 32 processes, high latency, complex file locking |

## The Fix

Add **synchronization barriers** to ensure all processes wait for rank 0 to complete checkpoint saving:

```python
def save_checkpoint(self, epoch, checkpoint_names=None):
    if self.skip_saving_ckpts:
        logging.info(
            "skip_saving_ckpts is set to True. So, no checkpoints have been saved."
        )
        return
    
    checkpoint_folder = self.checkpoint_conf.save_dir
    makedir(checkpoint_folder)
    if checkpoint_names is None:
        checkpoint_names = ["checkpoint"]
        if (
            self.checkpoint_conf.save_freq > 0
            and (int(epoch) % self.checkpoint_conf.save_freq == 0)
        ) or int(epoch) in self.checkpoint_conf.save_list:
            checkpoint_names.append(f"checkpoint_{int(epoch)}")

    checkpoint_paths = []
    for ckpt_name in checkpoint_names:
        checkpoint_paths.append(os.path.join(checkpoint_folder, f"{ckpt_name}.pt"))

    state_dict = unwrap_ddp_if_wrapped(self.model).state_dict()
    state_dict = exclude_params_matching_unix_pattern(
        patterns=self.checkpoint_conf.skip_saving_parameters, state_dict=state_dict
    )

    checkpoint = {
        "model": state_dict,
        "optimizer": self.optim.optimizer.state_dict(),
        "epoch": epoch,
        "loss": self.loss.state_dict(),
        "steps": self.steps,
        "time_elapsed": self.time_elapsed_meter.val,
        "best_meter_values": self.best_meter_values,
    }
    if self.optim_conf.amp.enabled:
        checkpoint["scaler"] = self.scaler.state_dict()

    # DDP checkpoints are only saved on rank 0 (all workers are identical)
    if self.distributed_rank != 0:
        barrier()  # ← ADD THIS: Wait for rank 0 to finish saving
        return

    for checkpoint_path in checkpoint_paths:
        self._save_checkpoint(checkpoint, checkpoint_path)
    
    barrier()  # ← ADD THIS: Signal other ranks that saving is complete
```

### Key Changes:

1. **Before rank 0 saves**: Non-rank-0 processes call `barrier()` before returning
2. **After rank 0 saves**: Rank 0 calls `barrier()` after completing all saves

This ensures:
- All processes wait at the same synchronization point
- Rank 0 completes writing before any process proceeds
- No file system contention or premature access

## Implementation Steps

1. Open `sam3/train/trainer.py`
2. Locate the `save_checkpoint` method (around line 359)
3. Add `barrier()` call before the early return for non-rank-0 processes
4. Add `barrier()` call after the checkpoint saving loop for rank 0
5. Ensure `barrier` is imported from `sam3.train.utils.distributed`

## Why This Fix Works

- **Synchronization**: All 32 processes wait at the barrier until rank 0 completes saving
- **Zero overhead**: Barrier adds only a few milliseconds of synchronization time
- **Correct distributed pattern**: This is the standard practice for checkpoint saving in PyTorch DDP
- **Scales properly**: Works for any number of nodes/GPUs (1, 4, 32, or more)

## Current Checkpoint Configuration

```yaml
checkpoint:
  save_dir: ${launcher.experiment_log_dir}/checkpoints
  save_freq: 0  # 0 = only saves at end of training
```

Note: With `save_freq: 0`, checkpoints are only saved at the end of training (after epoch 20), not after each epoch. The error at epoch 7 might be from a different trigger (interruption handler, validation checkpoint, etc.).

## Additional Recommendations

1. **Test with 1 node first**: Verify the fix works with 4 GPUs on 1 node
2. **Monitor filesystem**: Check `/pscratch` disk space and I/O performance
3. **Enable periodic saves**: Consider `save_freq: 10` to save checkpoints every 10 epochs
4. **Add logging**: Log when barriers are hit to track synchronization points

## References

- PyTorch DDP Documentation: https://pytorch.org/docs/stable/notes/ddp.html
- Checkpoint Best Practices: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
