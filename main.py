from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

# ───────── internal imports ───────────────────────────────────────── #
from core.utils import load_json, set_seed, ensure_dir
from data.data_utils import build_features
from graph.graph_builder import GraphBuilder
from train.trainer import Trainer  # supervised fine-tune trainer
from train.graphcl_trainer import GraphCLTrainer  # contrastive pre-trainer
from models.model import StockGNN
# ──────────────────────────────────────────────────────────────────── #


def _load_graph(cfg: dict) -> torch.Tensor:
    graph_fp = Path(cfg["data"]["processed"]) / "graph.pt"
    if not graph_fp.exists():
        raise FileNotFoundError(
            f"Graph file not found: {graph_fp}. "
            "Run with --mode preprocess first."
        )
    return torch.load(graph_fp)


def _instantiate_model(cfg: dict, load_pretrained: bool = True) -> StockGNN:
    rels = [
        "stock__corr__stock",
        "stock__dtw__stock",
        "stock__granger__stock",
        # ←  확장 여지 : influence, macro, text 관계 추가 가능
    ]
    model = StockGNN(cfg, rels)

    if load_pretrained and cfg.get("pretrain", {}).get("mode") == "graphcl":
        ckpt = Path(cfg["pretrain"]["save_path"])
        if ckpt.exists():
            print(f"[INFO] loading pretrained encoder from {ckpt}")
            state = torch.load(ckpt, map_location="cpu")
            missing, unexpected = model.load_state_dict(
                state, strict=False
            )  # heads will be missing
            print(f"[INFO] load_state_dict → missing={len(missing)} "
                  f"unexpected={len(unexpected)}")
        else:
            print(f"[WARN] no pretrained checkpoint found at {ckpt}")
    return model


def run_preprocess(cfg: dict) -> None:
    proc_dir = Path(cfg["data"]["processed"])
    ensure_dir(proc_dir)

    # 1) feature engineering
    feat_fp = proc_dir / "features.parquet"
    build_features(cfg["data"]["root"], feat_fp)

    # 2) heterogeneous graph
    gb = GraphBuilder(cfg)
    graph_fp = proc_dir / "graph.pt"
    gb.build(feat_fp, graph_fp)
    print(f"[DONE] preprocess → {graph_fp}")


def run_pretrain(cfg: dict) -> None:
    graph_data = _load_graph(cfg)
    model = _instantiate_model(cfg, load_pretrained=False)

    trainer = GraphCLTrainer(model, graph_data, cfg)
    trainer.fit(cfg["pretrain"]["epochs"], cfg["pretrain"]["save_path"])


def run_finetune(cfg: dict, test_only: bool = False) -> None:
    graph_data = _load_graph(cfg)
    model = _instantiate_model(cfg, load_pretrained=True)
    trainer = Trainer(model, graph_data, cfg)

    if test_only:
        ckpt = Path(cfg["artifacts_dir"]) / "best.pt"
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        val_rmse, _ = trainer.run_epoch(split="val")
        print(f"[RESULT] Test RMSE: {val_rmse**0.5:.6f}")
    else:
        # full fine-tune
        ensure_dir(cfg["artifacts_dir"])
        trainer.fit(cfg["train"]["epochs"], cfg["artifacts_dir"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=["preprocess", "pretrain", "train", "test"],
        help="Pipeline stage to run",
    )
    args = parser.parse_args()

    cfg = load_json("config/config.json")
    set_seed(cfg["seed"])
    torch.set_float32_matmul_precision("high")  # NVIDIA Ampere+

    mode = args.mode.lower()
    if mode == "preprocess":
        run_preprocess(cfg)
    elif mode == "pretrain":
        run_pretrain(cfg)
    elif mode == "train":
        run_finetune(cfg, test_only=False)
    elif mode == "test":
        run_finetune(cfg, test_only=True)
    else:  # pragma: no cover
        raise ValueError(f"Unknown mode '{mode}'")


if __name__ == "__main__":
    main()
