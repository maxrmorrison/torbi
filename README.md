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
```


#### `torbi.from_probabilities`

```python
```


#### `torbi.from_file`

```python
```


#### `torbi.from_file_to_file`

```python
```


#### `torbi.from_files_to_files`

```python
```


### Command-line interface

```python
```

**TODO**


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

`python -m penn.partition`

Randomly selects examples in the dataset for evaluation.


### Evaluate

### Evaluate

```
python -m penn.evaluate --config <config> --gpu <gpu>
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
