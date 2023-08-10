import torch.nn as nn
import utils

class ResBlock(nn.Module):
    def __init__(self, hidden):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3*output)

class Generator(nn.Module):
    def __init__(self, seq_len, hidden, noise_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(noise_size, hidden*seq_len)
        self.block = nn.Sequential(
            ResBlock(hidden),
            # ResBlock(hidden),
            # ResBlock(hidden),
            # ResBlock(hidden),
            # ResBlock(hidden),
        )
        self.conv1 = nn.Conv1d(hidden, 5, 1)
        self.hidden = hidden
        self.seq_len = seq_len

    def forward(self, noise):
        output = self.fc1(noise)
        output = output.reshape(-1, self.hidden, self.seq_len) # (BATCH_SIZE, DIM, SEQ_LEN)
        output = self.block(output)
        output = self.conv1(output)
        output = output.transpose(1, 2) # 第一二维交换
        shape = output.size()
        output = output.contiguous()    # 深拷贝（复制了一份数据）
        output = output.reshape(-1, 5)
        output = utils.gumbel_softmax(output, 0.5)
        return output.reshape(shape).transpose(1, 2) # (BATCH_SIZE, SEQ_LEN, len(charmap))

class Discriminator(nn.Module):
    def __init__(self, seq_len, hidden):
        super(Discriminator, self).__init__()
        self.conv1d = nn.Conv1d(5, hidden, 1)
        self.linear = nn.Linear(seq_len * hidden, 1)
        self.block = nn.Sequential(
            ResBlock(hidden),
            # ResBlock(hidden),
            # ResBlock(hidden),
            # ResBlock(hidden),
            # ResBlock(hidden),
        )
        self.hidden = hidden
        self.seq_len = seq_len

    def forward(self, input):
        output = self.conv1d(input)
        output = self.block(output)
        output = output.reshape(-1, self.seq_len*self.hidden)
        output = self.linear(output)
        return output