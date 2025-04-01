#include <vector>
#include <ATen/Operators.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>

// #include "viterbi.cpp"

#include <vector>

extern "C" {
    /* Creates a dummy empty _C module that can be imported from Python.
    The import from Python will load the .so consisting of this file
    in this extension, so that the TORCH_LIBRARY static initializers
    below are run. */
    PyObject* PyInit__C(void)
    {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_C",   /* name of module */
            NULL,   /* module documentation, may be NULL */
            -1,     /* size of per-interpreter state of the module,
                    or -1 if the module keeps state in global variables. */
            NULL,   /* methods */
        };
        return PyModule_Create(&module_def);
    }
}

namespace torbi {

// Defines the operator(s)
TORCH_LIBRARY(torbi, m) {
    m.def("viterbi_decode(Tensor observation, Tensor batch_frames, Tensor transition, Tensor initial) -> Tensor");
}

}  // namespace torbi
