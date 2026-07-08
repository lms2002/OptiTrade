import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        # 마지막 타임스텝의 출력만 사용
        out = self.fc(out[:, -1, :]) 
        return self.sigmoid(out)

class LSTMModel:
    def __init__(self, sequence_length=10, epochs=15, lr=0.01):
        self.seq_len = sequence_length
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean = None
        self.std = None

    def _create_sequences(self, X_val, y_val=None):
        xs, ys = [], []
        for i in range(len(X_val) - self.seq_len):
            xs.append(X_val[i:(i + self.seq_len)])
            if y_val is not None:
                ys.append(y_val[i + self.seq_len])
        return np.array(xs), (np.array(ys) if y_val is not None else None)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        # 딥러닝을 위한 Standard Scaling
        self.mean = X_train.mean(axis=0)
        self.std = X_train.std(axis=0) + 1e-8
        X_scaled = ((X_train - self.mean) / self.std).values
        y_val = y_train.values

        X_seq, y_seq = self._create_sequences(X_scaled, y_val)
        if len(X_seq) == 0:
            return

        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1).to(self.device)

        self.model = SimpleLSTM(input_dim=X_train.shape[1]).to(self.device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_t)
            loss = criterion(outputs, y_t)
            loss.backward()
            optimizer.step()
        logger.info(f"LSTM 학습 완료 (Device: {self.device})")

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """앙상블에 제공할 예측 확률(0~1) 반환"""
        if self.model is None: 
            return np.zeros(len(X_test))
            
        X_scaled = ((X_test - self.mean) / self.std).values
        
        # 길이를 맞추기 위해 앞부분에 seq_len 만큼의 더미 데이터 패딩
        pad = np.zeros((self.seq_len, X_test.shape[1]))
        X_padded = np.vstack([pad, X_scaled])
        
        X_seq, _ = self._create_sequences(X_padded)
        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_t).cpu().numpy().flatten()
        return preds

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X_test)
        return (probs > 0.5).astype(int)