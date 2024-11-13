# cmake_minimum_required(VERSION 3.21)

# find_package(PythonInterp REQUIRED)

# execute_process(
#     COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print('{}{}'.format(sys.version_info.major, sys.version_info.minor))"
#     OUTPUT_VARIABLE PYTHON_VERSION
#     OUTPUT_STRIP_TRAILING_WHITESPACE
# )

# execute_process(
#     COMMAND python -c "from torch.utils import cmake_prefix_path; \
#     print(cmake_prefix_path + '/Torch', end='')"
#     OUTPUT_VARIABLE Torch_DIR
# )

# include(ExternalProject)

# # environment variable
# set(ENV{HYHAL_PATH} "/opt/hyhal")
# set(ENV{ROCM_PATH} "/opt/dtk")
# set(ENV{CMAKE_INCLUDE_PATH} "$ENV{ROCM_PATH}/include:$ENV{HYHAL_PATH}/include:$ENV{ROCM_PATH}/llvm/include:$ENV{CMAKE_INCLUDE_PATH}")
# set(ENV{HIPCC_PATH} "$ENV{ROCM_PATH}/bin/hipcc")
# set(ENV{PATH} "$ENV{ROCM_PATH}/bin:$ENV{ROCM_PATH}/llvm/bin:$ENV{ROCM_PATH}/hip/bin:$ENV{ROPCM_PATH}/hip/hipify:$ENV{HYHAL_PATH}/bin:$ENV{PATH}")
# set(ENV{LD_LIBRARY_PATH} "$ENV{ROCM_PATH}/hip/lib:$ENV{ROCM_PATH}/llvm/lib:$ENV{ROCM_PATH}/lib:$ENV{ROCM_PATH}/lib64:$ENV{HYHAL_PATH}/lib:$ENV{HYHAL_PATH}/lib64:$ENV{LD_LIBRARY_PATH}")

# set(DCU_SUB_MODULE dcu_extension)
# set(DCU_SUB_MODULE_BINRAY_DIR "${PROJECT_BINARY_DIR}/${DCU_SUB_MODULE}")
# set(DCU_SUB_MODULE_BYPRODUCTS "${DCU_SUB_MODULE_BINRAY_DIR}/${DCU_SUB_MODULE}.cpython-${PYTHON_VERSION}-x86_64-linux-gnu.so")

# ExternalProject_Add(${DCU_SUB_MODULE}
#                     SOURCE_DIR "${PROJECT_SOURCE_DIR}/dlinfer/vendor/dcu/csrc"
#                     BINARY_DIR "${DCU_SUB_MODULE_BINRAY_DIR}"
#                     CMAKE_COMMAND "cmake"
#                     BUILD_BYPRODUCTS "${DCU_SUB_MODULE_BYPRODUCTS}"
#                     BUILD_ALWAYS ON
#                     USES_TERMINAL_BUILD ON
#                     USES_TERMINAL_INSTALL ON
#                     CMAKE_ARGS
#                       "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
#                       "-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
#                       "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
#                       "-DCMAKE_EXPORT_COMPILE_COMMANDS=${CMAKE_EXPORT_COMPILE_COMMANDS}"
#                       "-DTorch_DIR=${Torch_DIR}")

# install(
#   FILES ${DCU_SUB_MODULE_BYPRODUCTS}
#   DESTINATION ./dlinfer/vendor/dcu
# )
