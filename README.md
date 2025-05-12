<h1 align="center">Viterbi decoding in PyTorch</h1>
<div align="center">

[![PyPI](https://img.shields.io/pypi/v/torbi.svg)](https://pypi.python.org/pypi/torbi)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/torbi)](https://pepy.tech/project/torbi)

</div>


Highly parallelizable Viterbi decoding for CPU or GPU compute. Below are time benchmarks of our method relative to `librosa.sequence.viterbi`. We use 1440 states and ~20 million timesteps over ~40k files for benchmarking.

| Method  | Timesteps decoded per second |
| ------------- | ------------- |
| Librosa (1x cpu)| 208 |
| Librosa (16x cpu)| 1,382* |
| Proposed (1x cpu)| 171 |
| Proposed (16x cpu)| **2,240** |
| Proposed (1x a40 gpu, batch size 1)| **3,944,452** |
| Proposed (1x a40 gpu, batch size 512)| **692,160,422** |

*By default, `librosa.sequence.viterbi` uses one CPU thread. We use a Multiprocessing pool to parallelize.


## Table of contents

- [Installation](#installation)
- [Decoding](#decoding)
    * [Application programming interface](#application-programming-interface)
        * [`torbi.from_probabilities`](#torbifrom_probabilities)
        * [`torbi.from_file`](#torbifrom_file)
        * [`torbi.from_file_to_file`](#torbifrom_file_to_file)
        * [`torbi.from_files_to_files`](#torbifrom_files_to_files)
    * [Command-line interface](#command-line-interface)
- [Evaluation](#evaluation)
    * [Download](#download)
    * [Preprocess](#preprocess)
    * [Partition](#partition)
    * [Evaluate](#evaluate)
- [Citation](#citation)


## Installation

```bash
git clone git@github.com:maxrmorrison/torbi
pip install torbi/
```

If you receive an error message regarding mismatched CUDA versions, change the torch version in `pyproject.toml` to your currently installed version of `torch`.

To perform evaluation of the accuracy and speed of decoding methods, install `torbi` with the additional evaluation dependencies.

`pip install torbi[evaluate]`


## Decoding

### Application programming interface

```python
import torbi
import torch

# Time-varying categorical distribution to decode
observation = torch.tensor([
    [0.25, 0.5, 0.25],
    [0.25, 0.25, 0.5],
    [0.33, 0.33, 0.33]
]).unsqueeze(dim=0)

# Transition probabilities bewteen categories
transition = torch.tensor([
    [0.5, 0.25, 0.25],
    [0.33, 0.34, 0.33],
    [0.25, 0.25, 0.5]
])

# Initial category probabilities
initial = torch.tensor([0.4, 0.35, 0.25])

# Find optimal path using CPU compute
torbi.from_probabilities(
   observation,
   transition=transition,
   initial=initial,
   log_probs=False)

# Find optimal path using GPU compute
torbi.from_probabilities(
   observation,
   transition=transition,
   initial=initial,
   log_probs=False,
   gpu=0)
```


#### `torbi.from_probabilities`

```python
def from_probabilities(
    observation: torch.Tensor,
    batch_frames: Optional[torch.Tensor] = None,
    transition: Optional[torch.Tensor] = None,
    initial: Optional[torch.Tensor] = None,
    log_probs: bool = False,
    gpu: Optional[int] = None,
    num_threads: Optional[int] = 1
) -> torch.Tensor:
    """Decode a time-varying categorical distribution

    Arguments
        observation
            Time-varying categorical distribution
            shape=(batch, frames, states)
        batch_frames
            Number of frames in each batch item; defaults to all
            shape=(batch,)
        transition
            Categorical transition matrix; defaults to uniform
            shape=(states, states)
        initial
            Categorical initial distribution; defaults to uniform
            shape=(states,)
        log_probs
            Whether inputs are in (natural) log space
        gpu
            GPU index to use for decoding. Defaults to CPU.
        num_threads
            The number of threads to use for parallelized decoding

    Returns
        indices
            The decoded bin indices
            shape=(batch, frames)
    """
```


#### `torbi.from_file`

```python
def from_file(
    input_file: Union[str, os.PathLike],
    transition_file: Optional[Union[str, os.PathLike]] = None,
    initial_file: Optional[Union[str, os.PathLike]] = None,
    log_probs: bool = False,
    gpu: Optional[int] = None,
    num_threads: Optional[int] = 1
) -> torch.Tensor:
    """Decode a time-varying categorical distribution file

    Arguments
        input_file
            Time-varying categorical distribution file
            shape=(frames, states)
        transition_file
            Categorical transition matrix file; defaults to uniform
            shape=(states, states)
        initial_file
            Categorical initial distribution file; defaults to uniform
            shape=(states,)
        log_probs
            Whether inputs are in (natural) log space
        gpu
            GPU index to use for decoding. Defaults to CPU.
        num_threads
            The number of threads to use for parallelized decoding

    Returns
        indices
            The decoded bin indices
            shape=(frames,)
    """
```


#### `torbi.from_file_to_file`

```python
def from_file_to_file(
    input_file: Union[str, os.PathLike],
    output_file: Union[str, os.PathLike],
    transition_file: Optional[Union[str, os.PathLike]] = None,
    initial_file: Optional[Union[str, os.PathLike]] = None,
    log_probs: bool = False,
    gpu: Optional[int] = None,
    num_threads: Optional[int] = None
) -> None:
    """Decode a time-varying categorical distribution file and save

    Arguments
        input_file
            Time-varying categorical distribution file
            shape=(frames, states)
        output_file
            File to save decoded indices
        transition_file
            Categorical transition matrix file; defaults to uniform
            shape=(states, states)
        initial_file
            Categorical initial distribution file; defaults to uniform
            shape=(states,)
        log_probs
            Whether inputs are in (natural) log space
        gpu
            GPU index to use for decoding. Defaults to CPU.
        num_threads
            The number of threads to use for parallelized decoding
    """
```


#### `torbi.from_files_to_files`

```python
def from_files_to_files(
    input_files: List[Union[str, os.PathLike]],
    output_files: List[Union[str, os.PathLike]],
    transition_file: Optional[Union[str, os.PathLike]] = None,
    initial_file: Optional[Union[str, os.PathLike]] = None,
    log_probs: bool = False,
    gpu: Optional[int] = None,
    num_threads: Optional[int] = None
) -> None:
    """Decode time-varying categorical distribution files and save

    Arguments
        input_files
            Time-varying categorical distribution files
            shape=(frames, states)
        output_files
            Files to save decoded indices
        transition_file
            Categorical transition matrix file; defaults to uniform
            shape=(states, states)
        initial_file
            Categorical initial distribution file; defaults to uniform
            shape=(states,)
        log_probs
            Whether inputs are in (natural) log space
        gpu
            GPU index to use for decoding. Defaults to CPU.
        num_threads
            The number of threads to use for parallelized decoding
    """
```


### Command-line interface

```
usage: python -m torbi
    [-h]
    --input_files INPUT_FILES [INPUT_FILES ...]
    --output_files OUTPUT_FILES [OUTPUT_FILES ...]
    [--transition_file TRANSITION_FILE]
    [--initial_file INITIAL_FILE]
    [--log_probs]
    [--gpu GPU]
    [--num_threads NUM_THREADS]

arguments:
  --input_files INPUT_FILES [INPUT_FILES ...]
    Time-varying categorical distribution files
  --output_files OUTPUT_FILES [OUTPUT_FILES ...]
    Files to save decoded indices

optional arguments:
  -h, --help
    show this help message and exit
  --transition_file TRANSITION_FILE
    Categorical transition matrix file; defaults to uniform
  --initial_file INITIAL_FILE
    Categorical initial distribution file; defaults to uniform
  --log_probs
    Whether inputs are in (natural) log space
  --gpu GPU
    GPU index to use for decoding. Defaults to CPU.
  --num_threads NUM_THREADS
    The number of threads to use for parellelized CPU decoding
```


## Evaluation

### Download

`python -m torbi.data.download`

Downloads and decompresses the `daps` and `vctk` datasets used for evaluation.


### Preprocess

`python -m torbi.data.preprocess --gpu 0`

Preprocess the dataset to prepare time-varying categorical distributions for
evaluation. The distributions are pitch posteriorgrams produced by the `penn`
pitch estimator.


### Partition

`python -m torbi.partition`

Select all examples in dataset for evaluation.


### Evaluate

```
python -m torbi.evaluate --config <config> --gpu <gpu>
```

Evaluates the accuracy and speed of decoding methods. `<gpu>` is the GPU index.


## Citation

### IEEE
M. Morrison, C. Churchwell, N. Pruyne, and B. Pardo, "Fine-Grained and Interpretable Neural Speech Editing," Interspeech, September 2024.


### BibTex

```
@inproceedings{morrison2024fine,
    title={Fine-Grained and Interpretable Neural Speech Editing},
    author={Morrison, Max and Churchwell, Cameron and Pruyne, Nathan and Pardo, Bryan},
    booktitle={Interspeech},
    month={September},
    year={2024}
}
