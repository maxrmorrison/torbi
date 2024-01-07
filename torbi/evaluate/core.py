import json

import torch

import penn
import torchutil

import torbi


###############################################################################
# Evaluate
###############################################################################


def datasets(datasets, gpu=None):
    """Evaluate Viterbi decoding methods"""
    results = {}

    for dataset in datasets:

        # Reset benchmarking
        torchutil.time.reset()

        # Get evaluation stems
        with open(torbi.PARTITION_DIR / f'{dataset}.json') as file:
            stems = json.load(file)

        # Get input files
        input_files = [
            torbi.CACHE_DIR / dataset / f'{stem}.pt' for stem in stems]

        # Get location of reference outputs
        reference_files = [
            torbi.EVAL_DIR /
            dataset /
            'reference' /
            f'{stem}.pt' for stem in stems]

        # Run reference Librosa implementation if we haven't yet
        if not all(file.exists() for file in reference_files):
            torbi.reference.from_files_to_files(input_files, reference_files)

        # Get location to save output
        output_files = [
            torbi.EVAL_DIR /
            dataset /
            torbi.CONFIG /
            f'{stem}.pt' for stem in stems]

        # Run Viterbi decoding
        torbi.from_files_to_files(input_files, output_files, gpu=gpu)

        # Initialize metrics
        metrics = torbi.evaluate.Metrics()

        # Evaluate
        for predicted_file, target_file in zip(output_files, reference_files):
            predicted = torch.load(predicted)
            target = torch.load(target)
            metrics.update(predicted, target)

        # Get speed as real-time-factor (i.e., seconds decoded per second)
        seconds = penn.convert.frames_to_seconds(metrics.rpas[0].count)
        rtf = {kv[0]: seconds / kv[1] for kv in torchutil.time.results()}

        # Save
        results[dataset] = metrics() | {'rtf': rtf}
    with open(torbi.EVAL_DIR / 'results.json', 'w') as file:
        json.dump(results, file)
