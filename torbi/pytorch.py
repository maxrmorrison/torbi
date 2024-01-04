import torch
import numpy as np

def tforward(
    observation,
    transition,
    initial,
    posterior,
    memory,
    probability,
    frames,
    states
):
    t = 1
    tm1 = 0

    print('tforward started')

    # Add prior to first frame
    posterior[0] = observation[0] + initial
    print(posterior.device)

    # Forward pass
    while t < frames:
        # print(f'{t}/{frames}')

        # print('matadd')
        probability = posterior[tm1] + transition
        # print('done matadd')

        j = 0
        while j < states:

            # Get optimal greedy update from current state
            # s3 = 0
            # max_posterior = -torch.inf
            # while s3 < states:
            #     if probability[j, s3] > max_posterior:
            #         max_posterior = probability[j, s3]
            #         memory[t, j] = s3
            #     s3 = s3 + 1
            best_state = probability[j].argmax()
            memory[t, j] = best_state
            max_posterior = probability[j, best_state]

            # Update posterior distribution
            posterior[t, j] = observation[t, j] + max_posterior

            j = j + 1
        t = t + 1
        tm1 = tm1 + 1