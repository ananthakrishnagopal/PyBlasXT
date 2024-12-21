#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<iostream>

#include<cublas_v2.h>
#include<cuda_runtime.h>

#include<cublas_utils.h>

namespace py = pybind11;

/* library to perfrom matrix multiplication */


typedef py::array_t<float> farray;

farray multiply(farray fA,farray fB){
	py::buffer_info bA = fA.request();
	py::buffer_info bB = fB.request();
	
	// Assuming that A,B are 2-d arrays.

	//Create pointers in C++ for data
	float *A = static_cast<float*>(bA.ptr);
	float *B = static_cast<float*>(bB.ptr);
	
	//Get shapes of A,B
	size_t rA = bA.shape[0];
        size_t cA = bA.shape[1];

	size_t rB = bB.shape[0];
	size_t cB = bB.shape[1];

	if (cA != rB){
		std::cout<<"Shapes don't match."<<std::endl;
		std::exit(-1);
	}
	

	//Have numpy allocate memory. Access that pointer.
	py::array_t result = py::array_t<float>({rA,cB});
	
	py::buffer_info bC = result.request();	

	float *C = static_cast<float*>(bC.ptr);

	for(int i = 0;i<rA;i++){
		for(int j=0;j<cB;j++){
			float tmp = 0.0;
			for(int k=0;k<rB;k++){

				tmp+=A[i*rB + k]*B[k*rB+j];
			}
			C[i*cB+j] = tmp;
		}
	}
	
	return result;
}	
			
PYBIND11_MODULE(multiply,m){
	m.doc() = "testing ability to multiply in c++";
	m.def("multiply",&multiply,"multiplies two numpy arrays A,B");
}
