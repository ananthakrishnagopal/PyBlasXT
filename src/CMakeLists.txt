cmake_minimum_required(VERSION 3.18)

project(PyBlasXT
	VERSION 0.1
	LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)
include_directories(utils)


pybind11_add_module(multiply sgemm_cublas.cu)
target_link_libraries(multiply PRIVATE CUDA::cublas)
