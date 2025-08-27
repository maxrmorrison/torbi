import torch.utils.cpp_extension
from pathlib import Path

lib = torch.utils.cpp_extension.load(
    name='viterbi_mps',
    sources=[Path(__file__).parent / 'viterbi.mm'],
    extra_cflags=['-std=c++17'],
    is_python_module=False,
    # verbose=True
)