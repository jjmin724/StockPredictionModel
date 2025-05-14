import argparse, torch
from core.utils import load_json, set_seed, ensure_dir
from data.data_utils import build_features
from graph.graph_builder import GraphBuilder
from models.model import StockGNN
from train.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['preprocess','train','test'], required=True)
    args = parser.parse_args()
    cfg = load_json('config/config.json')
    set_seed(cfg['seed'])

    if args.mode=='preprocess':
        build_features("./data/raw", f"{cfg['data']['processed']}/features.parquet")
        gb = GraphBuilder(cfg)
        gb.build(f"{cfg['data']['processed']}/features.parquet",
                 f"{cfg['data']['processed']}/graph.pt")

    else:
        data = torch.load(f"{cfg['data']['processed']}/graph.pt")
        rels = ['stock__corr__stock','stock__dtw__stock','stock__granger__stock']
        model = StockGNN(cfg, rels)
        trainer = Trainer(model, data, cfg)
        if args.mode=='train':
            trainer.fit(cfg['train']['epochs'], cfg['artifacts_dir'])
        else:
            model.load_state_dict(torch.load(f"{cfg['artifacts_dir']}/best.pt"))
            loss,_ = trainer.run_epoch('val')
            print("Test RMSE:", loss**0.5)

if __name__=="__main__":
    main()
