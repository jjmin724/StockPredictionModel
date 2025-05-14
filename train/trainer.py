import torch, torch.nn as nn, torch.optim as optim, os, json
from torch_geometric.loader import NeighborLoader
from core.utils import save_json

class Trainer:
    def __init__(self, model, data, cfg):
        self.device = torch.device(cfg['device'])
        self.model = model.to(self.device)
        self.data = data.to(self.device)
        self.cfg = cfg
        self.opt = optim.AdamW(model.parameters(),
                               lr=cfg['train']['lr'],
                               weight_decay=cfg['train']['weight_decay'])
        self.loss_fn = nn.MSELoss()

    def run_epoch(self, mode='train'):
        self.model.train(mode=='train')
        self.opt.zero_grad()
        out, alpha = self.model(self.data)
        loss = self.loss_fn(out, self.data['stock'].y)
        if mode=='train':
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
        return loss.item(), alpha

    def fit(self, epochs, save_dir):
        best, patience = 1e9, 0
        hist=[]
        for ep in range(1,epochs+1):
            loss_tr,_ = self.run_epoch('train')
            loss_val,_ = self.run_epoch('val')
            hist.append({'epoch':ep,'train':loss_tr,'val':loss_val})
            if loss_val < best:
                best, patience = loss_val, 0
                torch.save(self.model.state_dict(), f"{save_dir}/best.pt")
            else:
                patience += 1
            if patience > self.cfg['train']['patience']:
                break
        save_json(hist, f"{save_dir}/loss_curve.json")
