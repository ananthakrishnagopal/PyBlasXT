# PyBlasXT
A Pybind11 Wrapper for CuBlasXt

#Usage

To build:

```
git clone <repo>
cd <repo>
mkdir build
cd build
cmake ..
make
```

To use in python:

goto build directory
`from src import PyBlasXT as pxt`


you can then call
`pxt.sgemm(a,b,devices)`
`pxt.dgemm(a,b,devices)`

note `devices` is a numpy array consisting of gpus to use `[0,1]` for the first and second device.

TODO:
create a wheel for easy install


