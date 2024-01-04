import librosa


def from_probabilities():
    """Perform reference Viterbi decoding"""

def from_file(input_file, transition, initial, log_probs):
    """Perform reference Viterbi decoding on a file"""

def from_file_to_file(input_file, output_file, transition, initial, log_probs):
    """Perform reference Viterbi decoding on a file and save"""


def from_files_to_files(input_files, output_files, transition, initial, log_probs):
    """Perform reference Viterbi decoding on many files and save"""
    for input_file, output_file in zip(input_files, output_files):
        from_file_to_file(input_file, output_file)
