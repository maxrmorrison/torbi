import torch
import torchutil

import torbi


###############################################################################
# Preprocess datasets
###############################################################################


@torchutil.notify('preprocess')
def datasets(datasets, gpu=None):
    """Preprocess a dataset"""
    try:
        import penn
    except ImportError:
        raise ImportError("penn is required for evaluation. Please install torbi with `torbi[evaluate]`")

    for dataset in datasets:

        # Get cache directory
        directory = torbi.CACHE_DIR / dataset

        # Get files
        audio_files = sorted(list(directory.rglob('*.wav')))

        # Preprocess pitch posteriorgrams
        for audio_file in audio_files:
            logits = []

            # Load audio file
            audio = penn.load.audio(audio_file)

            # Preprocess audio
            for frames in penn.preprocess(audio, center='half-hop'):

                # Copy to device
                frames = frames.to('cpu' if gpu is None else f'cuda:{gpu}')

                # Infer
                logits = penn.infer(frames).detach()

            # Concatenate results
            if isinstance(logits, torch.Tensor):
                logits = [logits]
            logits = torch.cat(logits, dim=0).squeeze(2)

            # Normalize
            logits = torch.nn.functional.log_softmax(logits, dim=1)

            # Save to cache
            torch.save(logits, audio_file.with_suffix('.pt'))
