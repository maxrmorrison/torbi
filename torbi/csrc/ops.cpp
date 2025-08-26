#include <vector>
#include <ATen/Operators.h>
#include <torch/library.h>

namespace torbi {

// Defines the operator(s)
TORCH_LIBRARY(torbi, m) {
    m.def("viterbi_decode(Tensor observation, Tensor batch_frames, Tensor transition, Tensor initial) -> Tensor");
}

}  // namespace torbi
