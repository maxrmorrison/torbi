#include <vector>
#include <ATen/Operators.h>
#include <torch/library.h>

// make the linker shut up about LNK2001 on Windows
//  (We don't care about it being importable as a python module)
#if defined(_MSC_VER)
extern "C" __declspec(dllexport) void* PyInit__C(void) {
    return nullptr;
}
#endif

namespace torbi {

// Defines the operator(s)
TORCH_LIBRARY(torbi, m) {
    m.def("viterbi_decode(Tensor observation, Tensor batch_frames, Tensor transition, Tensor initial) -> Tensor");
}

}  // namespace torbi
