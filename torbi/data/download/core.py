import shutil
from pathlib import Path

import torchaudio
import torchutil

import torbi


###############################################################################
# Download datasets
###############################################################################


@torchutil.notify('download')
def datasets(datasets):
    """Download datasets"""
    # Download and format daps dataset
    if 'daps' in datasets:
        daps()


def daps():
    """Download daps dataset"""
    torchutil.download.targz(
        'https://zenodo.org/record/4783456/files/daps-segmented.tar.gz?download=1',
        torbi.DATA_DIR)

    # Delete previous directory
    shutil.rmtree(promonet.DATA_DIR / 'daps', ignore_errors=True)

    # Rename directory
    data_directory = promonet.DATA_DIR / 'daps'
    shutil.move(
        promonet.DATA_DIR / 'daps-segmented',
        data_directory)

    # Get audio files
    audio_files = sorted(
        [path.resolve() for path in data_directory.rglob('*.wav')])
    text_files = [file.with_suffix('.txt') for file in audio_files]

    # Write audio to cache
    speaker_count = {}
    cache_directory = torbi.CACHE_DIR / 'daps'
    cache_directory.mkdir(exist_ok=True, parents=True)
    with torchutil.paths.chdir(cache_directory):

        # Iterate over files
        for audio_file, text_file in torchutil.iterator(
            zip(audio_files, text_files),
            'Formatting daps',
            total=len(audio_files)
        ):

            # Get speaker ID
            speaker = Path(audio_file.stem.split('_')[0])
            if speaker not in speaker_count:

                # Each entry is (index, count)
                speaker_count[speaker] = [len(speaker_count), 0]

            # Update speaker and get current entry
            speaker_count[speaker][1] += 1
            index, count = speaker_count[speaker]

            # Load audio
            audio, sample_rate = torchaudio.load(audio_file)

            # If audio is too quiet, increase the volume
            maximum = torch.abs(audio).max()
            if maximum < .35:
                audio *= .35 / maximum

            # Save at original sampling rate
            speaker_directory = cache_directory / f'{index:04d}'
            speaker_directory.mkdir(exist_ok=True, parents=True)
            output_file = Path(f'{count:06d}.wav')
            torchaudio.save(
                speaker_directory / output_file,
                audio,
                sample_rate)
            shutil.copyfile(
                text_file,
                (speaker_directory / output_file).with_suffix('.txt'))
