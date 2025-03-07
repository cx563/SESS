import torch
import torch.nn.functional as F


class InteractionSelfAttention(torch.nn.Module):
    def __init__(self, args):
        super(InteractionSelfAttention, self).__init__()
        self.args = args

    def forward(self, query, value, mask):
        attention_states = query
        attention_states_T = value
        attention_states_T = attention_states_T.permute([0, 2, 1])
        weights = torch.bmm(attention_states, attention_states_T)
        weights = weights.masked_fill(mask.unsqueeze(1).expand_as(weights)==0, float("-inf"))
        attention = F.softmax(weights, dim=2)
        return attention

