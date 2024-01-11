<h1 align="center">Viterbi decoding in PyTorch</h1>
<div align="center">

[![PyPI](https://img.shields.io/pypi/v/torbi.svg)](https://pypi.python.org/pypi/torbi)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/torbi)](https://pepy.tech/project/torbi)

</div>


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

`pip install torbi`

To perform evaluation of the accuracy and speed of decoding methods,
install the evaluation dependencies
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
])

# Transition probabilities bewteen categories
transition = torch.tensor([
    [0.5, 0.25, 0.25],
    [0.33, 0.34, 0.33],
    [0.25, 0.25, 0.5]
])

# Initial category probabilities
initial = torch.tensor([0.4, 0.35, 0.25])

# Find optimal path using CPU compute
torbi.decode(observation, transition, initial, log_probs=False)

# Find optimal path using GPU compute
torbi.decode(observation, transition, initial, log_probs=False, gpu=0)
```


#### `torbi.from_probabilities`

```python
def from_probabilities(
    observation: torch.Tensor,
    transition: Optional[torch.Tensor] = None,
    initial: Optional[torch.Tensor] = None,
    log_probs: bool = False,
    gpu: Optional[int] = None
) -> torch.Tensor:
    """Decode a time-varying categorical distribution

    Arguments
        observation
            Time-varying categorical distribution
            shape=(frames, states)
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

    Returns
        indices
            The decoded bin indices

```


#### `torbi.from_file`

```python
def from_file(
    input_file: Union[str, os.PathLike],
    transition_file: Optional[Union[str, os.PathLike]] = None,
    initial_file: Optional[Union[str, os.PathLike]] = None,
    log_probs: bool = False,
    gpu: Optional[int] = None
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
    gpu: Optional[int] = None
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
    gpu: Optional[int] = None
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
    """
```


### Command-line interface

```python
# TODO - usage
```

**TODO - docstring**


## Evaluation

### Download

`python -m torbi.data.download`

Downloads and uncompresses the `daps` dataset used for evaluation.


### Preprocess

`python -m torbi.data.preprocess`

Preprocess the dataset to prepare time-varying categorical distributions for
evaluation. The distributions are pitch posteriorgrams produced by the `penn`
pitch estimator.


### Partition

`python -m torbi.partition`

Randomly selects examples in the dataset for evaluation.


### Evaluate

```
python -m torbi.evaluate --config <config> --gpu <gpu>
```

Evaluates the accuracy and speed of decoding methods. `<gpu>` is the GPU index.


## Citation

### IEEE
M. Morrison, C. Churchwell, N. Pruyne, and B. Pardo, "Fine-Grained and Interpretable Neural Speech Editing," Submitted to International Conference on Machine Learning, <TODO - month> 2024.


### BibTex

```
@inproceedings{morrison2024fine,
    title={Fine-Grained and Interpretable Neural Speech Editing},
    author={Morrison, Max and Churchwell, Cameron and Pruyne, Nathan and Pardo, Bryan},
    booktitle={Submitted to International Conference on Machine Learning},
    month={TODO},
    year={2024}
}
