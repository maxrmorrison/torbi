<h1 align="center">Viterbi decoding in PyTorch</h1>
<div align="center">

[![PyPI](https://img.shields.io/pypi/v/torbi.svg)](https://pypi.python.org/pypi/torbi)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/torbi)](https://pepy.tech/project/torbi)

</div>

Current timings

Librosa:
{'setup': 0.0015919208526611328, 'forward-backward': 4.66046929359436, 'total': 4.6620612144470215}

Numpy:
{'setup': 0.007745981216430664, 'forward': 3.7035388946533203, 'backward': 0.0003383159637451172, 'total': 3.711623191833496}

Cython:
{'setup': 0.007486581802368164, 'forward': 13.393983125686646, 'backward': 0.0010607242584228516, 'total': 13.402530431747437}
