import torch
import torch.nn as nn


class LSTM_model(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 dropout):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # LSTM
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            dropout = dropout,
            batch_first = True
        )

        # Weight Initialization
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        self.dropout = nn.Dropout(dropout)

        # Shared Layer
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )

        # Heads

        # Direction (classification)
        self.fc_dir = nn.Linear(hidden_size // 2, 1)

        # Magnitude (regression)
        self.fc_mag = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        batch_size = x.size(0)

        # Initial hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM forward
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Take last time step
        last_hidden = torch.mean(lstm_out, dim=1) 

        out = self.shared(last_hidden) * 1.5

        # Outputs
        direction = self.fc_dir(out)   # logits
        magnitude = self.fc_mag(out)   # regression

        return direction, magnitude 