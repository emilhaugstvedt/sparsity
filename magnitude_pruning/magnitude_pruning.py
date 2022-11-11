import torch
import torch.nn as nn

from plain_net.PlainNet import PlainNet

class MagnitudePruningNet(PlainNet):
    def __init__(self, n_inputs, hidden_layers, n_outputs):
        super().__init__(n_inputs, hidden_layers, n_outputs)

    def prune(self, k):
        
