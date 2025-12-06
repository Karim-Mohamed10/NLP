# wrapper for pytorch-crf
from torchcrf import CRF
import torch.nn as nn
import torch

class CRFLayer(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, emissions, tags, mask):
        """
        emissions: (batch, seq_len, num_tags)
        tags: (batch, seq_len)
        mask: (batch, seq_len) bool
        returns: negative log likelihood loss
        """
        loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
        return loss

    def decode(self, emissions, mask):
        """
        returns list of lists with predicted tag indices
        """
        return self.crf.decode(emissions, mask=mask)
