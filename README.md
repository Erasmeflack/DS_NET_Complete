# Dual Siamese Network for Few-Shot SAR ATR

This project implements a Dual Siamese Network architecture designed for Few-Shot SAR Automatic Target Recognition (ATR) tasks. The framework supports both 5-way 1-shot and 5-way 5-shot experimental pipelines using modular components for training, testing, and evaluation.

## Project Structure

```
├── config/               # Experiment configuration files
├── data/                 # Dataset directory (MSTAR format)
├── models/               # Model components: Siamese branches & classifier
├── src/                  # Training & testing scripts
├── utils/                # Utilities: config parsing, logging, checkpointing
├── ckpt/                 # Saved model checkpoints
├── logs/                 # Training logs
├── plots/                # Visualizations and results
```

## Requirements

* Python 3.8+
* PyTorch >= 1.12
* torchvision
* numpy
* matplotlib
* tqdm
* easydict

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Setup

Prepare your dataset following the MSTAR structure:

```
./data/MSTAR/
├── train/
│   ├── Class1/
│   ├── Class2/
│   └── ...
└── test/
    ├── Class1/
    ├── Class2/
    └── ...
```

## Running 5-Way 1-Shot Experiment

### Train

```bash
python -m src.main --config config/5way_1shot.yaml
```

### Test (after training)

Update the YAML with `is_train: false` and run:

```bash
python -m src.main --config config/5way_1shot.yaml
```

## Running 5-Way 5-Shot Experiment

### Train

```bash
python src/main.py --config config/5way_5shot.yaml
```

### Test

Update the YAML with `is_train: false` and run:

```bash
python src/main.py --config config/5way_5shot.yaml
```

## Configuration Options

Key configurable options in `.yaml` files:

* `train_num_per_class`: Support set size per class (e.g., 1 or 5)
* `test_num_per_class`: Query set size per class
* `epochs`, `lr`, `train_patience`: Optimization settings
* `freeze_top`, `freeze_mid`, `freeze_bottom`: Selective sub-network freezing

## Output Directories

* `ckpt/`: Model checkpoints per experiment
* `logs/`: Training logs and metrics
* `plots/`: Visualizations (optional)

## Notes

* Ensure reproducibility with consistent `random_seed`
* Supports GPU acceleration via `use_gpu` flag
* Modular structure enables future extension for other few-shot settings

---

For questions or improvements, feel free to contribute or raise issues.
