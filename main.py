import argparse, torch
from core.utils import load_json, set_seed, ensure_dir
from data.data_utils import build_features
from graph.graph_builder import GraphBuilder
from models.model import StockGNN
from train.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['preprocess', 'train', 'test'], required=True)
    args = parser.parse_args()
    cfg = load_json('config/config.json')
    set_seed(cfg['seed'])
    if args.mode == 'preprocess':
        # 원시 데이터 전처리 및 그래프 생성
        build_features(cfg['data']['root'], f"{cfg['data']['processed']}/features.parquet")
        gb = GraphBuilder(cfg)
        gb.build(f"{cfg['data']['processed']}/features.parquet",
                 f"{cfg['data']['processed']}/graph.pt")
    else:
        # 그래프 데이터 불러오기 및 모델 초기화
        data = torch.load(f"{cfg['data']['processed']}/graph.pt")
        rels = [f"{src}__{rel}__{dst}" for src, rel, dst in data.edge_types]
        model = StockGNN(cfg, rels)
        trainer = Trainer(model, data, cfg)
        if args.mode == 'train':
            ensure_dir(cfg['artifacts_dir'])
            trainer.fit(cfg['train']['epochs'], cfg['artifacts_dir'])
        else:
            # 최적 모델 로드 후 검증(테스트) -> MSE, MAE, Sharpe Ratio 출력
            model.load_state_dict(torch.load(f"{cfg['artifacts_dir']}/best.pt"))
            model.eval()
            out, _ = model(data)
            mse = torch.nn.functional.mse_loss(out, data['stock'].y).item()
            mae = torch.nn.functional.l1_loss(out, data['stock'].y).item()
            out_np = out.detach().cpu().numpy()
            sharpe = float(out_np.mean() / (out_np.std() + 1e-9))
            print(f"Test MSE: {mse:.6f}, MAE: {mae:.6f}, Sharpe Ratio: {sharpe:.4f}")
            from core.utils import save_json
            save_json({"MSE": mse, "MAE": mae, "Sharpe": sharpe},
                      f"{cfg['artifacts_dir']}/test_metrics.json")

if __name__ == "__main__":
    main()
