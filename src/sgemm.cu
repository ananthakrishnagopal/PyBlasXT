#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<iostream>
#include<cublasXt.h>
#include<cublas_v2.h>
#include<cuda_runtime.h>

#include<cublas_utils.h>

namespace py = pybind11;

/* library to perfrom matrix multiplication */

typedef float elementType;
typedef py::array_t<float> farray;

farray multiply(farray fA,farray fB,farray fdevices){
	py::buffer_info bA = fA.request();
	py::buffer_info bB = fB.request();
	py::buffer_info bdevices = fdevices.request();
	
	// Assuming that A,B are 2-d arrays.

	//Create pointers in C++ for data
	elementType *A = static_cast<elementType*>(bA.ptr);
	elementType *B = static_cast<elementType*>(bB.ptr);
	int* devices = static_cast<int*>(bdevices.ptr);
	//Get shapes of A,B
	size_t M = bA.shape[0];
        size_t K = bA.shape[1];

	size_t N = bB.shape[1];

	
	

	//Have numpy allocate memory. Access that pointer.
	farray result = farray ({M,N});
	
	py::buffer_info bC = result.request();	

	elementType *C = static_cast<elementType*>(bC.ptr);
	
 	cublasXtHandle_t handle;

	CUBLAS_CHECK(cublasXtCreate(&handle));
	CUBLAS_CHECK(cublasXtDeviceSelect(handle,fdevices.size(),devices));
	elementType alpha = 1.0;
	elementType beta = 0.0;

	// performs (A@B).T
	//= B.T @ A.T
	//((A @ B ).T).T = (B.T@A.T).T
	CUBLAS_CHECK(
			cublasXtSgemm(handle,
				CUBLAS_OP_N,CUBLAS_OP_N,
				M,N,K,
				&alpha,
				B,N,
				A,K,
				&beta,
				C,N)
		    );



	CUBLAS_CHECK(cublasXtCreate(&handle));	
	return result;
}	
			
PYBIND11_MODULE(multiply,m){
	m.doc() = "testing ability to multiply in c++";
	m.def("multiply",&multiply,"multiplies two numpy arrays A,B");
}
