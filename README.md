# Octo-Small 1.5 LoRA Fine-tuning on BridgeData V2 — SimplerEnv WidowX Closed-loop Eval

机器学习课程 B 类项目：在 BridgeData V2 上对 `rail-berkeley/octo-small-1.5` 做轻量微调，在 SimplerEnv WidowX 的 T1/T2 两个任务上闭环评测 Base 与 Final。

## Results

Fixed 10 initial states per task; action-scale=0.8, gripper-threshold=0.5, 5 Hz control.

| Model | Trainable params | T1 spoon_on_towel | T2 eggplant_in_basket | B-SimplerScore |
|---|---|---|---|---|
| Base `octo-small-1.5` step 300000 | — | 0/10 | 3/10 (30%) | 15.0 |
| Final-A, action-head only (step 750) | 27/366 kernels | 0/10 | 0/10 | 0.0 |
| Final-B, full fine-tune (step 750) | 366/366 kernels | 0/10 | 0/10 | 0.0 |

Per-episode records: `results/b_formal/episodes.csv`. Representative videos: `results/b_formal/representative/`.

## Model

- Base: [`rail-berkeley/octo-small-1.5`](https://huggingface.co/rail-berkeley/octo-small-1.5), step 300000
- Final-A: frozen transformer, trained diffusion action head for 750 steps
- Final-B: all parameters trainable for 750 steps
- Both trained on BridgeData V2 (`bridge_dataset`) with surface-placement / container-placement filtering

## Reproduce

```bash
# install
pip install -r requirements.txt

# Base / Final-A / Final-B eval (independent processes per model)
python evaluate.py --model-name Base --model-type octo-small \
  --ckpt-path <base_path> --task widowx_spoon_on_towel --episodes 10 \
  --eval-config configs/eval_config.json --output-dir results/b_formal/base \
  --action-scale 0.8 --gripper-threshold 0.5 --record-video

# fine-tune
bash train.sh
```

Final checkpoint loading: see `checkpoint_link.txt`.

## Layout

```
├── README.md
├── requirements.txt
├── train.sh / train_config.py
├── evaluate.py
├── configs/
│   ├── eval_config.json            # fixed initial states (frozen)
│   ├── checkpoint_metadata_head_only/
│   └── checkpoint_metadata_full/
├── scripts/
│   ├── octo_bridge_policy.py        # SimplerEnv ↔ Octo adapter
│   ├── task_config.py
│   ├── placement_filter.py
│   └── summarize_results.py
├── logs/
│   ├── train_log_head_only.txt
│   └── train_log_full.txt
├── results/
│   └── b_formal/
│       ├── episodes.csv
│       ├── results.csv
│       ├── base/  final_a_head_only/  final_b_full/
│       └── representative/
├── checkpoint_link.txt
└── video_links.md
```
