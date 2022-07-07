# Codelet Mining
This repository provides an algorithm for mining for codelets to vectorize sparse matrix code. 


## Build
You will need a C++ compiler (tested for GCC) and CMake to build this repository. MKL library is also needed for comparing the performance.  

To build the repository from source:
```
git clone --recursive https://github.com/sympiler/codelet_mining.git
cd codelet_mining
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/opt/intel/oneapi/mkl/2021.2.0/lib/intel64/;/opt/intel/oneapi/mkl/2021.2.0/include/" ..
make 
```


## Run a demo
There a set of demos to test different kerneks. You will need a matrix file stored as matrix market format (matrix.mtx). Then for example you test one of demos with
```
./demo/spmv_demo -m matrix.mtx -n SPMV -s CSR -t 8 

```
which runs different implementations of SpMV CSR for the given matrix.


