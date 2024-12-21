cmake_minimum_required(VERSION 3.18)

project(PyBlasXT
	VERSION 0.1
	LANGUAGES C CXX CUDA)

#add_subdirectory(extern)
add_subdirectory(extern/pybind11)
add_subdirectory(src)

