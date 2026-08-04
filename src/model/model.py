import torch
import torch.nn as nn

class MatchupClassifier(nn.Module):
    '''simple feedforward network that predicts a matchup outcome from difference features

    a single hidden layer MLP with dropout, outputting a single logit for the
    probability of a win (or upset, depending on how labels were built)
    '''
    def __init__(self, config) -> None:
        '''builds the network from the experiment config

        Parameters
        ----------
        config : dict
            config of the experiment, to get the feature count, hidden layer size, and dropout rate
        '''
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''runs a forward pass through the network

        Parameters
        ----------
        x : torch.Tensor
            batch of feature vectors shape (B, F)

        Returns
        -------
        torch.Tensor
            batch of raw logits (pre-sigmoid) shape (B, 1)
        '''
        return self.network(x)