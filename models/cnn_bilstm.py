# models/cnn_bilstm.py

import torch.nn as nn

class CNNBiLSTM(nn.Module):
    def __init__(self, input_dim=39, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(64, 128, bidirectional=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)         # (batch, features, time) → for Conv1D
        x = self.conv(x)               # Conv1D expects (B, C, T)
        x = x.permute(2, 0, 1)         # (time, batch, features) → for LSTM
        x, _ = self.lstm(x)            # BiLSTM outputs (T, B, 2*hidden)
        return self.fc(x[-1])          # Take last timestep
