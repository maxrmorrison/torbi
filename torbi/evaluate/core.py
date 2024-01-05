import json

import torchutil

import torbi


###############################################################################
# Evaluate
###############################################################################


def datasets(datasets):
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

            # Run reference
            torbi.reference.from_files_to_files(input_files, output_files)

        # Get location to save output
        output_files = [
            torbi.EVAL_DIR /
            dataset /
            torbi.CONFIG /
            f'{stem}.pt' for stem in stems]

        # Run Viterbi decoding
        torbi.from_files_to_files(input_files, output_files)

        # Initialize metrics
        metrics = torbi.evaluate.Metrics()

        # Evaluate
        for predicted_file, target_file in zip(output_files, reference_files):
            predicted = torch.load(predicted)
            target = torch.load(target)
            metrics.update(predicted, target)

        # Save
        results[dataset] = {'speed': torchutil.time.results()} | metrics()
    with open(torbi.EVAL_DIR / 'results.json', 'w') as file:
        json.dump(results, file)
