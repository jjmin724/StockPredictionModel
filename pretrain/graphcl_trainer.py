"""
GraphCL self-supervised trainer
-------------------------------
Saves encoder weights for later downstream fine-tuning.
"""
import json
from pathlib import Path

import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

from graph.augmentations import GraphAugmentor
from models.model import GraphCLModel
from data.data_utils import load_dataset   # existing util


class GraphCLTrainer:
    def __init__(self, cfg_path: str):
        cfg = json.load(open(cfg_path))
        self.cfg = cfg
        self.device = torch.device(cfg["device"])

        dataset = load_dataset(cfg["data"]["root"])
        self.loader = DataLoader(dataset,
                                 batch_size=cfg["train"]["batch_size"],
                                 shuffle=True)

        # assume node feature size already known
        in_dim = dataset.num_node_features
        hid_dim = cfg["train"]["hidden_dim"] if "hidden_dim" in cfg["train"] else 128
        proj_dim = cfg["pretrain"]["proj_dim"]

        self.model = GraphCLModel(in_dim, hid_dim, proj_dim).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=cfg["train"]["lr"])
        self.augmentor = GraphAugmentor(cfg["pretrain"]["augment_strength"])

    # ----------- contrastive loss -----------
    def info_nce(self, z1, z2, temperature: float = 0.5):
        B = z1.size(0)
        sim = torch.exp(torch.mm(z1, z2.t()) / temperature)
        positives = torch.diag(sim)
        denom = sim.sum(dim=1)
        loss = -torch.log(positives / (denom + 1e-8)).mean()
        return loss

    # ----------- main loop -----------
    def train(self):
        epochs = self.cfg["pretrain"]["epochs"]
        for epoch in range(1, epochs + 1):
            total_loss = 0.
            for batch in self.loader:
                batch = batch.to(self.device)
                # two stochastic views
                view1 = self.augmentor(batch)
                view2 = self.augmentor(batch)
                z1 = self.model(view1)
                z2 = self.model(view2)
                loss = self.info_nce(z1, z2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch.num_graphs

            avg = total_loss / len(self.loader.dataset)
            if epoch % 10 == 0:
                print(f"[Pre-Train] Epoch {epoch:03d} | Loss={avg:.4f}")

        save_p = Path(self.cfg["pretrain"]["save_path"])
        save_p.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self.model.encoder.state_dict(), save_p)
        print("✓ Pre-trained encoder saved ->", save_p)
