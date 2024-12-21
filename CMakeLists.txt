cmake_minimum_required(VERSION 3.18)
set(SKBUILD_PROJECT_NAME PyBlasXT)
project(${SKBUILD_PROJECT_NAME}
	VERSION 0.1
	LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
#add_subdirectory(extern)
add_subdirectory(extern/pybind11)
add_subdirectory(src)

