import torch
import torch.nn as nn

class LSTM_Predictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.1):
        """
        LSTM-based Predictor for sequential features.
        :param input_dim: int, feature dimension per time step (e.g., 7)
        :param hidden_dim: int, hidden dimension of LSTM
        :param num_layers: int, number of LSTM layers
        :param dropout: float, dropout rate
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        :param x: Tensor, shape (batch_size, seq_len, input_dim) == (X, 5, 7)
        :return: Tensor, shape (batch_size, 1)
        """
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_dim)
        last_hidden = lstm_out[:, -1, :]  # 取最后一个时间步的输出
        out = self.dropout(last_hidden)
        return self.fc(out)  # (batch_size, 1)
