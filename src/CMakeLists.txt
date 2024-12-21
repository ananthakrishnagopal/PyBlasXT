cmake_minimum_required(VERSION 3.8)

project(PyBlasXT
	VERSION 0.1
	LANGUAGES C CXX CUDA)

pybind11_add_module(multiply sgemm.cpp)
