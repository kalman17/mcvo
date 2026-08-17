# mcvo — image-only self-supervised visual odometry

Transformer on frozen DINOv2 features that predicts relative camera pose (and per-pixel
uncertainty) from images alone. Trained without labels: cached UniDepth depth and UniMatch
flow supervise AnyCam's uncertainty-weighted flow-reprojection loss; nothing but images is
needed at inference.

- `model.py` — `MCVO`: backbone + alternating temporal/spatial attention blocks with a
  per-frame camera token → poses `[B, N, 1, 4, 4]` (cam_i → cam_{i+1}, last = identity) and
  uncertainty maps.
- `loss.py` — self-supervised flow-reprojection loss, identity-motion reference, and two
  optional terms that were evaluated and *not* adopted (teacher-pose distillation, epipolar
  Sampson loss).
- `train.py` — trainer on the preprocessed AnyCam-style dataset (`experiments/datasets/`),
  deterministic split, per-epoch validation, resumable checkpoints.

Released weights (E3, 8-frame training, 6 epochs): https://huggingface.co/thekman17/mcvo

```bash
# train
PYTHONPATH=. python mcvo/train.py --data_dir /path/to/preprocessed --save_dir runs/mcvo \
    --backbone facebook/dinov2-base --d_model 640 --depth 10 --heads 8 --max_ahead 7 \
    --batch_size 4 --lr 1.5e-4 --epochs 6
# evaluate (window protocol, same harness as the thesis results)
PYTHONPATH=. python experiments/honest_benchmark.py --run_name mcvo_eval \
    --datasets sintel,tumrgbd,kitti --models mcvo:runs/mcvo/checkpoints/epoch_0006.pt
```
