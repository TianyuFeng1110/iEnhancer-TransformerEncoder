import torch
import math
import torch.nn as nn
import numpy as np

class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_heads, dropout):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(vocab_size, num_heads)
        self.addnorm1 = AddNorm(vocab_size, dropout)
        self.ffn = PositionWiseFFN(vocab_size, hidden_size, vocab_size)
        self.addnorm2 = AddNorm(vocab_size, dropout)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.addnorm1(x, attn_output)
        x = self.addnorm2(x, self.ffn(x))
        return x

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class model(nn.Module):
    def __init__(self, vocab_size, mer_num, num_hiddens, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout, mer_num)
        self.transformer_encoder = nn.Sequential(
            TransformerEncoderBlock(num_hiddens, hidden_size=2048, num_heads=8, dropout=dropout),
            TransformerEncoderBlock(num_hiddens, hidden_size=2048, num_heads=8, dropout=dropout),
            TransformerEncoderBlock(num_hiddens, hidden_size=2048, num_heads=8, dropout=dropout),
            TransformerEncoderBlock(num_hiddens, hidden_size=2048, num_heads=8, dropout=dropout),
            TransformerEncoderBlock(num_hiddens, hidden_size=2048, num_heads=8, dropout=dropout),
            TransformerEncoderBlock(num_hiddens, hidden_size=2048, num_heads=8, dropout=dropout),
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(mer_num*num_hiddens, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pos_encoding(self.embedding(x) * math.sqrt(self.num_hiddens))
        x = self.transformer_encoder(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x