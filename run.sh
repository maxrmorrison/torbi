# Run evaluation of Viterbi decoding methods

# Args
# $1 - index of GPU to use

# Setup data for evaluation
python -m torbi.data.download --datasets daps
python -m torbi.data.preprocess --datasets daps --gpu $1
python -m torbi.partition --datasets daps

# Evaluate
python -m torbi.evaluate --datasets daps --gpu $1
