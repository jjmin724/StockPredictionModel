import torch
import torch.nn as nn
import torch.optim as optim
from models.encoders import TimeSeriesEncoder   # 기존 모델 내부 encoder 사용

class PretrainingPipeline:
    """
    사전학습 파이프라인 (sequence / static 지원)
    학습 완료 후 encoder state_dict 파일(.pt) 저장
    """
    def __init__(self, mode='sequence', input_dim=2, hidden_dim=64,
                 lr=1e-3, epochs=100):
        assert mode in ('sequence', 'static')
        self.mode = mode
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if mode == 'sequence':
            self.encoder = TimeSeriesEncoder(input_dim, hidden_dim).to(self.device)
        else:  # static
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ).to(self.device)

        self.predictor = nn.Linear(hidden_dim, 1).to(self.device)
        params = list(self.encoder.parameters()) + list(self.predictor.parameters())
        self.optimizer = optim.Adam(params, lr=lr)
        self.criterion = nn.MSELoss()

    def train(self, X, y):
        """
        X: [N, T, F] (sequence) or [N, F] (static)
        y: [N]
        """
        X, y = X.to(self.device), y.to(self.device)
        best, best_sd = float('inf'), None
        for epoch in range(1, self.epochs + 1):
            self.encoder.train(); self.predictor.train()
            if self.mode == 'sequence':
                pred = self.predictor(self.encoder(X))
            else:
                pred = self.predictor(self.encoder(X))
            loss = self.criterion(pred.squeeze(), y)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            self.optimizer.step()

            if loss.item() < best:
                best, best_sd = loss.item(), {
                    k: v.cpu() for k, v in self.encoder.state_dict().items()
                }
        fname = ("pretrained_stock_encoder.pt"
                 if self.mode == 'sequence' else
                 "pretrained_static_encoder.pt")
        torch.save(best_sd, fname)
        return best
