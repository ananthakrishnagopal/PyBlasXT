#include<pybind11/pybind11.h>
#include<iostream>
namespace py = pybind11;

void hello()
{
	std::cout<<"hello world~!"<<std::endl;
	return ;
}

PYBIND11_MODULE(hello,m){
	m.doc() = "hello world example";
	m.def("hello",&hello,"A function that prints hello");
}
