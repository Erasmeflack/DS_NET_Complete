# utils/config_io.py
import os
import copy
import yaml
import pprint
from typing import Any, Dict, Iterable, Optional


# ---------------------------
# Basic YAML helpers
# ---------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[config_io] YAML not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"[config_io] YAML root must be a mapping (dict): {path}")
    return data


def dump_yaml(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


# ---------------------------
# Dot-path overrides
#   e.g. overrides={"lr":"0.0005", "augmentation.flip":"false"}
# ---------------------------
def apply_overrides(cfg: Dict[str, Any], overrides: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    if not overrides:
        return cfg

    def parse_scalar(v: str):
        # Try to coerce types: bool, int, float; fallback to string
        vl = v.lower()
        if vl in ("true", "false"):
            return vl == "true"
        try:
            if "." in v:
                return float(v)
            return int(v)
        except ValueError:
            return v

    out = copy.deepcopy(cfg)
    for k, v in overrides.items():
        parts = k.split(".")
        d = out
        for p in parts[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]
        d[parts[-1]] = parse_scalar(v)
    return out


# ---------------------------
# Validation / Normalization
# ---------------------------
_VALID_BRANCHES = {"S", "D", "C"}
_VALID_PROTO_METRICS = {"euclidean", "cosine"}
_VALID_AGG = {"mean", "max"}

def _normalize(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)

    # Device
    use_gpu = bool(cfg.get("use_gpu", True))
    cfg["device"] = "cuda" if (use_gpu and _cuda_available()) else "cpu"

    # DataLoader perf flags (defaults)
    cfg.setdefault("num_workers", 4)
    cfg.setdefault("pin_memory", True)
    cfg.setdefault("persistent_workers", True)
    cfg.setdefault("prefetch_factor", 2)

    # Episode controls
    cfg.setdefault("episodes_per_epoch", 200)
    cfg.setdefault("val_episodes", 200)

    # Aggregation
    agg = cfg.get("agg_method", "mean")
    if agg not in _VALID_AGG:
        raise ValueError(f"[config_io] agg_method must be one of {sorted(_VALID_AGG)}, got: {agg}")

    # Branch controls for relation module (training)
    used_branches = cfg.get("relation_used_branches", ["S", "D", "C"])
    if not isinstance(used_branches, Iterable):
        raise ValueError("[config_io] relation_used_branches must be a list/iterable of {'S','D','C'}.")
    used_branches = set(used_branches)
    if not used_branches.issubset(_VALID_BRANCHES):
        raise ValueError(f"[config_io] relation_used_branches must be subset of {_VALID_BRANCHES}, got: {used_branches}")
    cfg["relation_used_branches"] = sorted(list(used_branches))

    # Branch controls for TEST ablations (fallback to train setting)
    test_branches = cfg.get("test_relation_used_branches", None)
    if test_branches is None:
        cfg["test_relation_used_branches"] = cfg["relation_used_branches"]
    else:
        if not isinstance(test_branches, Iterable):
            raise ValueError("[config_io] test_relation_used_branches must be a list/iterable of {'S','D','C'}.")
        test_branches = set(test_branches)
        if not test_branches.issubset(_VALID_BRANCHES):
            raise ValueError(f"[config_io] test_relation_used_branches must be subset of {_VALID_BRANCHES}, got: {test_branches}")
        cfg["test_relation_used_branches"] = sorted(list(test_branches))

    # A2 prototype loss knobs
    if cfg.get("use_proto_loss", False):
        metric = cfg.get("proto_metric", "euclidean")
        if metric not in _VALID_PROTO_METRICS:
            raise ValueError(f"[config_io] proto_metric must be one of {_VALID_PROTO_METRICS}, got: {metric}")
        cfg.setdefault("lambda_proto", 0.3)
        cfg.setdefault("proto_normalize", True)
        cfg.setdefault("proto_temperature", 1.0)

    # Loss weights defaults
    cfg.setdefault("lambda_rel", 1.0)
    cfg.setdefault("lambda_sim", 0.2)
    cfg.setdefault("lambda_dissim", 0.2)
    cfg.setdefault("lambda_cls", 0.25)

    # AMP / grad clip
    cfg.setdefault("amp", False)
    cfg.setdefault("grad_clip", 0.0)

    # Augmentation flag (bool or dict)
    aug = cfg.get("augmentation", True)
    if isinstance(aug, dict):
        cfg["_augment_enabled"] = any(bool(v) for v in aug.values())
    else:
        cfg["_augment_enabled"] = bool(aug)

    # Core paths + exp dir
    exp_name = cfg.get("experiment_name", "default_exp")
    cfg["exp_dir"] = os.path.join(cfg.get("logs_dir", "./logs"), f"exp_{exp_name}")
    cfg["ckpt_dir_exp"] = os.path.join(cfg.get("ckpt_dir", "./ckpt"), f"exp_{exp_name}")
    cfg["plot_dir_exp"] = os.path.join(cfg.get("plot_dir", "./plots"), f"exp_{exp_name}")

    # Common logs
    cfg["loss_log_path"] = os.path.join(cfg["exp_dir"], "loss.csv")
    cfg["val_log_path"]  = os.path.join(cfg["exp_dir"], "val.csv")
    cfg["_config_snapshot"] = os.path.join(cfg["exp_dir"], "config.frozen.yaml")

    return cfg


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------
# Entry points
# ---------------------------
def load_config(yaml_path: str, overrides: Optional[Dict[str, str]] = None, make_dirs: bool = True) -> Dict[str, Any]:
    """
    Load a YAML config, apply optional CLI overrides (dot-path),
    normalize/validate, and create experiment directories + snapshot.

    Returns:
        cfg (dict): normalized configuration dictionary
    """
    raw = load_yaml(yaml_path)
    cfg = apply_overrides(raw, overrides)
    cfg = _normalize(cfg)

    if make_dirs:
        _ensure_dirs([
            cfg["exp_dir"],
            cfg["ckpt_dir_exp"],
            cfg["plot_dir_exp"],
        ])
        # snapshot the normalized config for reproducibility
        dump_yaml(cfg, cfg["_config_snapshot"])
        # initialize unified loss log (if not exists)
        _init_loss_csv(cfg["loss_log_path"])
        # initialize val log (if not exists)
        _init_val_csv(cfg["val_log_path"])

    return cfg


def pretty_print_cfg(cfg: Dict[str, Any]) -> str:
    """
    Return a pretty string of key training knobs and paths.
    """
    keys = [
        "experiment_name", "device", "n_way", "k_shot", "q_query",
        "epochs", "episodes_per_epoch", "val_episodes",
        "lr", "optimizer", "lr_scheduler",
        "lambda_rel", "lambda_sim", "lambda_dissim", "lambda_cls",
        "use_proto_loss", "lambda_proto", "proto_metric", "proto_normalize", "proto_temperature",
        "relation_used_branches", "test_relation_used_branches", "agg_method",
        "_augment_enabled",
        "exp_dir", "ckpt_dir_exp", "plot_dir_exp",
    ]
    picked = {k: cfg.get(k) for k in keys if k in cfg}
    return pprint.pformat(picked, indent=2, width=100)


# ---------------------------
# File initializers
# ---------------------------
def _ensure_dirs(dirs: Iterable[str]) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _init_loss_csv(path: str) -> None:
    if not os.path.exists(path):
        with open(path, "w") as f:
            # unified loss file with columns:
            f.write("epoch,sim_loss,dis_loss,cls_loss,rel_loss,uni_loss\n")


def _init_val_csv(path: str) -> None:
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("epoch,acc,avg_loss\n")


# ---------------------------
# Utilities for callers
# ---------------------------
def get_exp_dirs(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Return key experiment directories."""
    return {
        "exp_dir": cfg["exp_dir"],
        "ckpt_dir": cfg["ckpt_dir_exp"],
        "plot_dir": cfg["plot_dir_exp"],
    }


def get_log_paths(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Return unified log paths (loss + val)."""
    return {
        "loss_log": cfg["loss_log_path"],
        "val_log": cfg["val_log_path"],
    }
