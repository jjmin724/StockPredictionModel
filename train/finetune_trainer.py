"""
Finetuning trainer that optionally loads pre-trained encoder weights.
"""
import json
from pathlib import Path
import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

from models.model import GNNEncoder          # <-- downstream encoder
from data.data_utils import load_dataset


class FineTuneTrainer:
    def __init__(self, cfg_path: str):
        cfg = json.load(open(cfg_path))
        self.cfg = cfg
        self.device = torch.device(cfg["device"])

        dataset = load_dataset(cfg["data"]["root"])
        # simple split
        n = len(dataset)
        train_set = dataset[: int(0.8*n)]
        test_set = dataset[int(0.8*n):]

        self.train_loader = DataLoader(train_set,
                                       batch_size=cfg["train"]["batch_size"],
                                       shuffle=True)
        self.test_loader = DataLoader(test_set,
                                      batch_size=cfg["train"]["batch_size"])
        in_dim = dataset.num_node_features
        hid_dim = cfg["train"]["hidden_dim"] if "hidden_dim" in cfg["train"] else 128

        self.encoder = GNNEncoder(in_dim, hid_dim).to(self.device)
        # load weights if opted
        ckpt = cfg["pretrain"].get("save_path")
        if cfg["pretrain"]["mode"] == "graphcl" and ckpt and Path(ckpt).is_file():
            print("• Loading pre-trained weights from", ckpt)
            self.encoder.load_state_dict(torch.load(ckpt, map_location=self.device))

        self.cls_head = torch.nn.Linear(hid_dim, 1).to(self.device)
        params = list(self.encoder.parameters()) + list(self.cls_head.parameters())
        self.optim = Adam(params, lr=cfg["train"]["lr"])

    def train(self):
        for epoch in range(self.cfg["train"]["epochs"]):
            self.encoder.train(); self.cls_head.train()
            tot = 0.; correct = 0
            for batch in self.train_loader:
                batch = batch.to(self.device)
                h = self.encoder(batch.x, batch.edge_index, batch.batch)
                logits = self.cls_head(h).squeeze()
                loss = F.binary_cross_entropy_with_logits(logits, batch.y.float())

                self.optim.zero_grad(); loss.backward(); self.optim.step()
                tot += loss.item() * batch.num_graphs
                preds = (logits.sigmoid() > 0.5).long()
                correct += (preds == batch.y).sum().item()

            train_acc = correct / len(self.train_loader.dataset)
            print(f"[FineTune] Epoch {epoch:03d} | Loss={tot:.4f} | Acc={train_acc:.4f}")

    def evaluate(self):
        self.encoder.eval(); self.cls_head.eval()
        correct = 0
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                logits = self.cls_head(
                    self.encoder(batch.x, batch.edge_index, batch.batch)
                ).squeeze()
                preds = (logits.sigmoid() > 0.5).long()
                correct += (preds == batch.y).sum().item()
        acc = correct / len(self.test_loader.dataset)
        print("Test-set Accuracy =", acc)
