# https://github.com/mttk/rnn-classifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import time
from temp import train_iter, valid_iter, test_iter, input_feature_size

from enum import Enum


class RNN(Enum):
    LSTM = 0
    GRU = 1


class INFERENCE(Enum):
    CLASSIFICATION = 0
    REGRESSION = 1

rnn_type = RNN.LSTM
inference = INFERENCE.REGRESSION


HIDDEN_SIZE = 128
N_LAYERS = 2
BIDIRECTIONAL = False
N_CLASSES = 2
CLIP = 2


def seed_everything(seed, cuda=False):
    # Set the random seed manually for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, nlayers=1, dropout=0., bidirectional=True):
        super(Encoder, self).__init__()
        self.bidirectional = bidirectional

        if rnn_type is RNN.LSTM:
            self.rnn = nn.LSTM(
              embedding_dim, hidden_dim, nlayers, batch_first=True,
              dropout=dropout, bidirectional=bidirectional
            )
        elif rnn_type is RNN.GRU:
            self.rnn = nn.GRU(
              embedding_dim, hidden_dim, nlayers, batch_first=True,
              dropout=dropout, bidirectional=bidirectional
            )
        else:
            raise ValueError()

    def forward(self, input, hidden=None):
        return self.rnn(input, hidden)


class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim) # Scaled Dot Product

    def forward(self, query, keys, values):
        # Query = [BxH]       B: Batch Size, Q: Hidden Size
        # Keys = [BxSxH]      B: Batch Size, S: Seq. Size, H: Hidden Size
        # Values = [BxSxH]    B: Batch Size, S: Seq. Size, H: Hidden Size
        # Outputs = episode_reward:[BxS], attention_value:[BxH]

        # Here we assume q_dim == k_dim (dot product attention)

        query = query.unsqueeze(1)                    # [BxH] -> [Bx1xH]
        # keys = keys.transpose(0, 1).transpose(1, 2)   # [BxSxH] -> [BxHxS]
        keys = keys.transpose(1, 2)                   # [BxSxH] -> [BxHxS]
        episode_reward = torch.bmm(query, keys)                # [Bx1xH]x[BxHxS] -> [Bx1xS]
        episode_reward = F.softmax(episode_reward.mul_(self.scale), dim=2)    # scale & normalize

        # values = values.transpose(1, 2)             # [BxSxH] -> [BxHxS]
        attention_value = torch.bmm(episode_reward, values).squeeze(1)    # [Bx1xS]x[BxSxH] -> [BxH]

        return episode_reward.unsqueeze(1), attention_value


class NullEmbedding(nn.Module):
    def __init__(self):
        super(NullEmbedding, self).__init__()

    def forward(self, input):
        return input


class SelfAttentionRNNClassifier(nn.Module):
    def __init__(self, embedding, encoder, attention, hidden_dim, num_classes):
        super(SelfAttentionRNNClassifier, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.attention = attention
        self.decoder = nn.Linear(hidden_dim, num_classes)

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size))

    def forward(self, input):
        outputs, hidden = self.encoder(self.embedding(input))
        if isinstance(hidden, tuple):    # LSTM
            hidden = hidden[1]    # take the cell state

        if self.encoder.bidirectional:    # need to concat the last 2 hidden layers
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden = hidden[-1]

        episode_reward, attention_value = self.attention(hidden, outputs, outputs)
        logits = self.decoder(attention_value)
        return logits, episode_reward


class SelfAttentionRNNRegressor(nn.Module):
    def __init__(self, embedding, encoder, attention, hidden_dim):
        super(SelfAttentionRNNRegressor, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.attention = attention
        self.decoder = nn.Linear(hidden_dim, 1)

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size))

    def forward(self, input):
        outputs, hidden = self.encoder(self.embedding(input))
        # output: [32, 18, 128]

        if isinstance(hidden, tuple):    # LSTM
            hidden = hidden[1]    # take the cell state
            # hidden: [2, 32, 128]

        if self.encoder.bidirectional:    # need to concat the last 2 hidden layers
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden = hidden[-1]
            # hidden: [32, 128]

        episode_reward, attention_value = self.attention(hidden, outputs, outputs)

        pred_value = self.decoder(attention_value).squeeze(1)  # [B:1] --> [B]

        return pred_value, episode_reward


def train(model, data, optimizer, criterion):
    model.train()

    t = time.time()
    total_loss = 0

    for batch_num, batch in enumerate(data):
        x, y = batch

        if inference is INFERENCE.CLASSIFICATION:
            logits, _ = model(x)
            loss = criterion(logits.view(-1, N_CLASSES), y)
        elif inference is INFERENCE.REGRESSION:
            pred_value, _ = model(x)
            loss = criterion(pred_value, y)
        else:
            raise ValueError()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()

        total_loss += float(loss)

        print("[Batch]: {}/{} in {:.5f} seconds".format(
            batch_num + 1, len(data), time.time() - t), end='\r', flush=True
        )
        t = time.time()

    print()
    print("[Train Loss]: {:.5f}".format(total_loss / len(data)))

    return total_loss / len(data)


def evaluate(model, data, criterion, type='Valid'):
    model.eval()

    t = time.time()
    total_loss = 0
    with torch.no_grad():
        for batch_num, batch in enumerate(data):
            x, y = batch

            if inference is INFERENCE.CLASSIFICATION:
                logits, _ = model(x)
                total_loss += float(criterion(logits.view(-1, N_CLASSES), y))

            elif inference is INFERENCE.REGRESSION:
                pred_value, _ = model(x)
                total_loss += float(criterion(pred_value, y))

            else:
                raise ValueError()

            print("[Batch]: {}/{} in {:.5f} seconds".format(
                batch_num + 1, len(data), time.time() - t), end='\r', flush=True
            )
            t = time.time()

    print()
    print("[{} loss]: {:.5f}".format(type, total_loss / len(data)))
    return total_loss / len(data)


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    seed_everything(seed=1337, cuda=cuda)
    device = torch.device("cpu") if not cuda else torch.device("cuda:0")

    encoder = Encoder(
        embedding_dim=input_feature_size,
        hidden_dim=HIDDEN_SIZE,
        nlayers=N_LAYERS,
        dropout=0.0,
        bidirectional=BIDIRECTIONAL
    )

    embedding = NullEmbedding()

    attention_dim = HIDDEN_SIZE if not BIDIRECTIONAL else 2 * HIDDEN_SIZE
    attention = Attention(attention_dim, attention_dim, attention_dim)

    if inference is INFERENCE.CLASSIFICATION:
        model = SelfAttentionRNNClassifier(embedding, encoder, attention, attention_dim, N_CLASSES)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    elif inference is INFERENCE.REGRESSION:
        model = SelfAttentionRNNRegressor(embedding, encoder, attention, attention_dim)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    else:
        raise ValueError()

    model.to(device)
    MAX_EPOCH = 100

    try:
        best_valid_loss = None

        for epoch in range(1, MAX_EPOCH + 1):
            print("*************** EPOCH: {0} ***************".format(epoch))
            train(model, train_iter, optimizer, criterion)
            loss = evaluate(model, valid_iter, criterion, type='Valid')

            if not best_valid_loss or loss < best_valid_loss:
                best_valid_loss = loss
            print()

    except KeyboardInterrupt:
        print("[Ctrl+C] Training stopped!")

    loss = evaluate(model, test_iter, criterion, type='Test')
