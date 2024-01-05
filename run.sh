# Run evaluation of Viterbi decoding methods

# Args
# $1 - index of GPU to use

# Setup data for evaluation
python -m torbi.data.download
python -m torbi.data.preprocess --gpu $1
python -m torbi.partition

# Evaluate
python -m torbi.evaluate --gpu $1
