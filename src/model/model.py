import torch.nn as nn

class MatchupClassifier(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        input_dim = len(config['features']['columns'])
        hidden_dims = config['model']['hidden_layers']
        dropout = config['model']['dropout']

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, 1)
        )

    def forward(self, x):
        return self.network(x)

