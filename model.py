import math
import torch
import torch.nn as nn
from torchinfo import summary

class ConvBlock(nn.Sequential):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int = 1, 
        padding: int = 0, 
        activation: nn.Module = nn.LeakyReLU
    ):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            activation(),
            nn.BatchNorm2d(out_channels),
        )
class InceptionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Sequential(
            ConvBlock(32, 64, kernel_size=(1, 1), padding='same'),
            ConvBlock(64, 64, kernel_size=(3, 1), padding='same')
        )
        self.branch2 = nn.Sequential(
            ConvBlock(32, 64, kernel_size=(1, 1), padding='same'),
            ConvBlock(64, 64, kernel_size=(5, 1), padding='same')
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            ConvBlock(32, 64, kernel_size=(1, 1), padding='same')
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)

class DeepLOB(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.lstm_hidden_size = 64
        self.lstm_num_layers = 1
        # Convolutiona
        self.conv1 = nn.Sequential(
            ConvBlock(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            ConvBlock(32, 32, kernel_size=(4, 1)),
            ConvBlock(32, 32, kernel_size=(4, 1)),
        )
        self.conv2 = nn.Sequential(
            ConvBlock(32, 32, kernel_size=(1, 2), stride=(1, 2), activation=nn.Tanh),
            ConvBlock(32, 32, kernel_size=(4, 1), activation=nn.Tanh),
            ConvBlock(32, 32, kernel_size=(4, 1), activation=nn.Tanh),
        )
        self.conv3 = nn.Sequential(
            ConvBlock(32, 32, kernel_size=(1, 10)),
            ConvBlock(32, 32, kernel_size=(4, 1)),
            ConvBlock(32, 32, kernel_size=(4, 1)),
        )
        # Inception Module
        self.inception = InceptionModule()
        # LSTM
        self.lstm = nn.LSTM(input_size=192, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)
        self.fc = nn.Linear(self.lstm_hidden_size, 3)

    def forward(self, x):
        h0 = torch.zeros(self.lstm_num_layers, x.size(0), self.lstm_hidden_size).to(self.device)
        c0 = torch.zeros(self.lstm_num_layers, x.size(0), self.lstm_hidden_size).to(self.device)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.inception(x) # -> x(batch_size, 192, 82, 1)
        x = x.permute(0, 2, 1, 3)
        x = x.squeeze(-1)
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, :x.shape[1], :].detach()
        return self.dropout(x)

class LOBFormer(nn.Module):
    def __init__(
        self,
        input_dim=40,
        seq_len=100,
        d_model=128,
        nhead=8,
        num_layers=4,
        dropout=0.1
    ):
        super().__init__()
        # In time series data, the input is already a sequence of vectors.
        # So rather than using an embedding layer, we use a linear layer to project the input to the embedding space.
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            activation='gelu',
            # norm_first
            batch_first=True  # Important for time series data
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.output_head = nn.Linear(d_model, 3)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.dropout(x[:, -1, :]) # Use latest token(snapshot)
        return torch.softmax(self.output_head(x), dim=1)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn = DeepLOB(device=device)
    transformer = LOBFormer()

    summary(cnn, (1, 1, 100, 40))
    summary(transformer, (100, 100, 40))
