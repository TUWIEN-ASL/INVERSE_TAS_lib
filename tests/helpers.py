"""Shared test helpers — imported by test files in the same package."""

import torch


class MockBatchGenerator:
    """Returns synthetic tensors so training tests need no real data files."""

    def __init__(self, num_classes, feat_dim=16, seq_len=50, n_videos=2):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.list_of_examples = [f"vid{i}" for i in range(n_videos)]
        self.index = 0

    def has_next(self):
        return self.index < len(self.list_of_examples)

    def reset(self):
        self.index = 0

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index : self.index + batch_size]
        self.index += batch_size
        B = len(batch)
        inp = torch.randn(B, self.feat_dim, self.seq_len)
        tgt = torch.randint(0, self.num_classes, (B, self.seq_len))
        mask = torch.ones(B, self.num_classes, self.seq_len)
        return inp, tgt, mask


class DummyBackbone(torch.nn.Module):
    """Minimal backbone compatible with HASR's train.py interface.

    backbone(x, mask) must return something where result[-1] is (B, num_classes, T).
    """

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = torch.nn.Conv1d(input_dim, num_classes, 1)

    def forward(self, x, mask):
        return [self.fc(x)]
