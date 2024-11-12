// 2024 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include <torch/extension.h>

#include "cache.h"
#include "ops.h"

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // vLLM custom ops
    pybind11::module ops = m.def_submodule("ops", "vLLM custom operators");

    ops.def("reshape_and_cache_new",
            &reshape_and_cache_new,
            "reshape_and_cache_new(Tensor key, Tensor value,"
            "                  Tensor! key_cache, Tensor! value_cache,"
            "                  Tensor slot_mapping,"
            "                  str kv_cache_dtype,"
            "                  float kv_scale,"
            "                  float v_scale) -> ()");
}
