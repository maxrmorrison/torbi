import json

import torch
import torchutil

import torbi


###############################################################################
# Evaluate
###############################################################################


def datasets(datasets, gpu=None, num_threads=1):
    """Evaluate Viterbi decoding methods"""
    # Cache transition matrix

    try:
        import penn
    except ImportError:
        raise ImportError("penn is required for evaluation. Please install torbi with `torbi[evaluate]`")

    if not torbi.PITCH_TRANSITION_MATRIX.exists():
        xx, yy = torch.meshgrid(
            torch.arange(penn.PITCH_BINS),
            torch.arange(penn.PITCH_BINS),
            indexing='ij')
        bins_per_octave = penn.OCTAVE / penn.CENTS_PER_BIN
        max_octaves_per_frame = \
            penn.MAX_OCTAVES_PER_SECOND * penn.HOPSIZE / penn.SAMPLE_RATE
        max_bins_per_frame = max_octaves_per_frame * bins_per_octave + 1
        transition = torch.clip(max_bins_per_frame - (xx - yy).abs(), 0)
        transition = transition / transition.sum(dim=1, keepdims=True)
        torch.save(transition, torbi.PITCH_TRANSITION_MATRIX)

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

        # get transition file
        transition_file = torbi.PITCH_TRANSITION_MATRIX

        # Get location of reference outputs
        if torbi.COMPARE_WITH_REFERENCE:
            reference_files = [
                torbi.EVAL_DIR /
                dataset /
                'reference' /
                f'{stem}.pt' for stem in stems]

            # Create parent directories
            (torbi.EVAL_DIR / dataset / 'reference').mkdir(
                parents=True,
                exist_ok=True)
            for file in reference_files:
                file.parent.mkdir(exist_ok=True)

            # Run reference Librosa implementation if we haven't yet
            if not all(file.exists() for file in reference_files):
                torbi.reference.from_files_to_files(
                    input_files,
                    reference_files,
                    transition_file=transition_file,
                    log_probs=True,
                    num_threads=num_threads)

        else: # Compare with non-chunked
            reference_files = [
                torbi.EVAL_DIR /
                dataset /
                'torbi' /
                f'{stem}.pt' for stem in stems]

        # Get location to save output
        output_files = [
            torbi.EVAL_DIR /
            dataset /
            torbi.CONFIG /
            f'{stem}.pt' for stem in stems]

        # Create parent directories
        (torbi.EVAL_DIR / dataset / torbi.CONFIG).mkdir(parents=True, exist_ok=True)
        for file in output_files:
            file.parent.mkdir(exist_ok=True)

        # Run Viterbi decoding
        torbi.from_files_to_files(
            input_files,
            output_files,
            transition_file=transition_file,
            log_probs=True,
            gpu=gpu,
            num_threads=num_threads)

        # Initialize metrics
        metrics = torbi.evaluate.Metrics()

        # Evaluate
        for predicted_file, target_file in zip(output_files, reference_files):
            predicted = torch.load(predicted_file, map_location='cpu')
            target = torch.load(target_file, map_location='cpu')
            metrics.update(predicted, target)

        # Get speed as real-time-factor (i.e., seconds decoded per second)
        seconds = penn.convert.frames_to_seconds(metrics.rpas[0].count)
        rtf = {
            key: seconds / value
            for key, value in torchutil.time.results().items()}

        # Save
        results[dataset] = metrics() | {'rtf': rtf}
    with open(torbi.EVAL_DIR / f'{torbi.CONFIG}.json', 'w') as file:
        json.dump(results, file)
