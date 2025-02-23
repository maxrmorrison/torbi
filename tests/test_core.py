import torch
import numpy as np
import time

import torbi

def test_decode():
    """Viterbi decoding test"""
    observation = torch.tensor([
        [0.25, 0.5, 0.25],
        [0.25, 0.25, 0.5],
        [0.33, 0.33, 0.33]
    ]).unsqueeze(dim=0)
    transition = torch.tensor([
        [0.5, 0.25, 0.25],
        [0.33, 0.34, 0.33],
        [0.25, 0.25, 0.5]
    ])
    initial = torch.tensor([0.4, 0.35, 0.25])
    bins = torbi.from_probabilities(
        observation=observation,
        transition=transition,
        initial=initial,
        log_probs=False)
    assert (bins == torch.tensor([1, 2, 2])).all()

def test_cuda():
    """Viterbi decoding test with CUDA"""
    observation = torch.tensor([
        [0.25, 0.5, 0.25],
        [0.25, 0.25, 0.5],
        [0.33, 0.33, 0.33]
    ]).unsqueeze(dim=0).to('cuda:0')
    transition = torch.tensor([
        [0.5, 0.25, 0.25],
        [0.33, 0.34, 0.33],
        [0.25, 0.25, 0.5]
    ]).to('cuda:0')
    initial = torch.tensor([0.4, 0.35, 0.25]).to('cuda:0')
    bins = torbi.from_probabilities(
        observation=observation,
        transition=transition,
        initial=initial,
        log_probs=False,
        gpu=0)
    assert (bins == torch.tensor([1, 2, 2]).to('cuda:0')).all()
