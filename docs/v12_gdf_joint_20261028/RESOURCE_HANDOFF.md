# Two-GPU resource handoff

User-confirmed sequence, 28 October 2026:

1. Keep the current three local-scope controls running unchanged through training and native VAL8 evaluation.
2. After all three endpoints and their files are complete and their training processes have exited, report completion to the user.
3. Hold the new Gaussian revision while the user stops the instance, resizes it from three GPUs to two, and restarts it. Retain its disks and checkpoints.
4. Verify the actual two-GPU inventory, unchanged source and parent checkpoint, and exited prior controller identity. Then explicitly release the resource gate and resume the single waiting controller.
5. Execute one shared new Gaussian-head warmup and calibration, then run the matched full-S AR / AR_GDF suffixes on one GPU each.

The resource gate defaults to closed. Completing the old experiments alone cannot launch the new revision. Both explicit operator release and exactly two visible GPUs are required. No target overlay, CUDA check or new training starts before release.

Waiting-controller recovery archives the prior lock and state only after confirming a dead controller and no revision stages. An exclusive resume claim prevents duplicate recovery. A controller that has already begun revision work cannot use this waiting-only recovery path.

This scheduling change does not alter the fixed method, data, losses, sampling, seed or successful-update budgets. No extra experiment is added to fill a third GPU. Fifteen CPU regression checks cover the completion barrier, closed resource gate and waiting-only recovery.
