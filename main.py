import argparse
import torch

from core.utils import (
    load_json,
    save_json,
    set_seed,
    ensure_dir,
)
from data.data_utils import build_features, load_pretrain_dataset
from graph.graph_builder import GraphBuilder
from models.model import StockGNN
from train.trainer import Trainer
from train.pretraining import PretrainingPipeline

# NEW ──────────────────────────────────────────────────────────────
from data_collect import collect_data  # data 수집 패키지 호출

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["collect", "preprocess", "pretrain", "train", "test"],
        required=True,
        help=(
            "collect: 원천 데이터 수집\n"
            "preprocess: 특징 & 그래프 생성\n"
            "pretrain: 시계열 인코더 사전학습\n"
            "train: 모델 학습\n"
            "test: 저장된 모델 평가"
        ),
    )
    args = parser.parse_args()

    cfg = load_json("config/config.json")
    set_seed(cfg["seed"])

    # ---------------------- (0) 데이터 수집 ----------------------
    if args.mode == "collect":
        collect_data()  # data_collect 패키지 실행
        return

    # ---------------------- (1) 데이터 전처리 --------------------
    if args.mode == "preprocess":
        build_features(
            f"{cfg['data']['root']}/prices",
            f"{cfg['data']['processed']}/features.parquet",
        )
        gb = GraphBuilder(cfg)
        gb.build(
            f"{cfg['data']['processed']}/features.parquet",
            f"{cfg['data']['processed']}/graph.pt",
        )
        return

    # ---------------------- (2) 사전학습 ------------------------
    if args.mode == "pretrain":
        # 사용자가 load_pretrain_dataset을 구현했다고 가정
        X_seq, y = load_pretrain_dataset(cfg)  # Tensor [N, T, F], [N]
        pretrainer = PretrainingPipeline(
            mode="sequence",
            input_dim=X_seq.size(-1),
            hidden_dim=cfg["model"]["in_dim"],
            lr=cfg["train"]["lr"],
            epochs=cfg["train"]["epochs"],
        )
        best_loss = pretrainer.train(X_seq, y)
        print(f"[Pretrain] best MSE = {best_loss:.6f}")
        return

    # ---------------------- (3) 모델 학습 & 테스트 ---------------
    data = torch.load(f"{cfg['data']['processed']}/graph.pt")
    rels = [f"{s}__{r}__{d}" for s, r, d in data.edge_types]

    model = StockGNN(cfg, rels)

    # 사전학습된 가중치 로드 (존재 시)
    try:
        state_dict = torch.load("pretrained_stock_encoder.pt")
        model.stock_encoder.load_state_dict(state_dict, strict=False)
        print("[Info] Pretrained encoder loaded.")
    except FileNotFoundError:
        print("[Warn] No pretrained encoder found – training from scratch.")

    trainer = Trainer(model, data, cfg)
    if args.mode == "train":
        ensure_dir(cfg["artifacts_dir"])
        trainer.fit(cfg["train"]["epochs"], cfg["artifacts_dir"])
    elif args.mode == "test":
        model.load_state_dict(torch.load(f"{cfg['artifacts_dir']}/best.pt"))
        model.eval()
        out, _ = model(data)

if __name__ == "__main__":
    main()
