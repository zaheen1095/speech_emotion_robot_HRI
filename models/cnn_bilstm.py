import torch.nn as nn
import torch

class CNNBiLSTM(nn.Module):
    def __init__(self, input_dim=39, num_classes=2,use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        
        # Enhanced CNN Block
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),  # Keep temporal dim
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Enhanced BiLSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            bidirectional=True,
            batch_first=True,
            num_layers=2,
            dropout=0.3
        )
        
        # Attention Layer
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classifier
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # Input shape: (batch, time, features)
        x = x.permute(0, 2, 1)  # (batch, features, time)
        
        # CNN
        x = self.conv(x)  # (batch, channels, time)
        x = x.permute(0, 2, 1)  # (batch, time, channels)
        
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, time, 2*hidden)
        
        # Attention
         # ... conv -> lstm_out (B, T, 256)
        if self.use_attention:
            attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (B, T, 1)
            context = torch.sum(attn_weights * lstm_out, dim=1)                   # (B, 256)
        else:
            context = torch.mean(lstm_out, dim=1)       # (B, 256) mean-pool
        # attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        # context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Classification
        return self.fc(context)
