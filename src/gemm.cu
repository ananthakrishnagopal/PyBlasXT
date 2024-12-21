#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<iostream>
#include<cublasXt.h>
#include<cublas_v2.h>
#include<cuda_runtime.h>

#include<cublas_utils.h>

namespace py = pybind11;

/* library to perfrom matrix multiplication */

typedef py::array_t<float> farray;
farray sgemm(farray fA,farray fB,py::array_t<int> fdevices){
	py::buffer_info bA = fA.request();
	py::buffer_info bB = fB.request();
	py::buffer_info bdevices = fdevices.request();
	
	// Assuming that A,B are 2-d arrays.

	//Create pointers in C++ for data
	float *A = static_cast<float*>(bA.ptr);
	float *B = static_cast<float*>(bB.ptr);
	int* devices = static_cast<int*>(bdevices.ptr);
	//Get shapes of A,B
	size_t M = bA.shape[0];
        size_t K = bA.shape[1];

	size_t N = bB.shape[1];

	
	

	//Have numpy allocate memory. Access that pointer.
	farray result = farray ({M,N});
	
	py::buffer_info bC = result.request();	

	float *C = static_cast<float*>(bC.ptr);
	
 	cublasXtHandle_t handle;

	CUBLAS_CHECK(cublasXtCreate(&handle));
	CUBLAS_CHECK(cublasXtDeviceSelect(handle,fdevices.size(),devices));
	float alpha = 1.0;
	float beta = 0.0;

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



typedef py::array_t<double> darray;

darray dgemm(darray fA,darray fB,py::array_t<int> fdevices){
	py::buffer_info bA = fA.request();
	py::buffer_info bB = fB.request();
	py::buffer_info bdevices = fdevices.request();
	
	// Assuming that A,B are 2-d arrays.

	//Create pointers in C++ for data
	double *A = static_cast<double*>(bA.ptr);
	double *B = static_cast<double*>(bB.ptr);
	int* devices = static_cast<int*>(bdevices.ptr);
	//Get shapes of A,B
	size_t M = bA.shape[0];
        size_t K = bA.shape[1];

	size_t N = bB.shape[1];

	
	

	//Have numpy allocate memory. Access that pointer.
	darray result = darray ({M,N});
	
	py::buffer_info bC = result.request();	

	double *C = static_cast<double*>(bC.ptr);
	
 	cublasXtHandle_t handle;

	CUBLAS_CHECK(cublasXtCreate(&handle));
	CUBLAS_CHECK(cublasXtDeviceSelect(handle,fdevices.size(),devices));
	double alpha = 1.0;
	double beta = 0.0;

	// performs (A@B).T
	//= B.T @ A.T
	//((A @ B ).T).T = (B.T@A.T).T
	CUBLAS_CHECK(
			cublasXtDgemm(handle,
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
			
PYBIND11_MODULE(PyBlasXT,m){
	m.doc() = "testing ability to multiply in c++";
	m.def("dgemm",&dgemm,"multiplies two numpy arrays A,B  - dgemm");
	m.def("sgemm",&sgemm,"multiplies two numpy arrays A,B  - sgemm");
}
