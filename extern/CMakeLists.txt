cmake_minimum_required(VERSION 3.18)

include(FetchContent)

# ------------------------------------------------------------------------------
# PyBind11 

FetchContent_Declare(
	pybind11
	GIT_REPOSITORY https://github.com/pybind/pybind11.git
	GIT_TAG v2.13.6)

FetchContent_MakeAvailable(pybind11)
