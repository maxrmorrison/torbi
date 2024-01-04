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
    ])
    transition = torch.tensor([
        [0.5, 0.25, 0.25],
        [0.33, 0.34, 0.33],
        [0.25, 0.25, 0.5]
    ])
    initial = torch.tensor([0.4, 0.35, 0.25])
    bins = torbi.decode(observation, transition, initial, log_probs=False)
    assert (bins == torch.tensor([1, 2, 2])).all()

def test_argmax():
    # array = np.random.uniform(size=(int(4*8e7-5),))
    array = np.random.uniform(size=(int(6000),))

    m = array.max()

    start = time.time()
    o = array.argmax()
    end = time.time()
    print('numpy argmax:', end-start, "seconds")

    # test argmax implementation
    idx = torbi.argmax(array)
    assert o == idx, f"{array[o]} vs {array[idx]}"

    # test parallel argmax implementation
    idx = torbi.parallel_argmax(array)
    assert o == idx, f"{array[o]} vs {array[idx]}"

    # test argmax using indices implementation
    idx = torbi.argmax_opt(array)
    assert o == idx, f"{array[o]} vs {array[idx]}"
