# GPU Kernels for Bhattacharya distance between two point sets

GPU implementation of a simplyfied Bhattacharya distance between two point sets that explicitly takes
anisotropic localization uncertainty into account.


## Example usage

The code currently consists of a single class which can be instantiated to allocate
resources on the GPU. The number you pass to the constructor is the largest possible
size of the point set you wish to compute on (this is simply used for memmory allocations).

The GPUExpDist object performs expensive memory allocations and deallocations and
as such it should be reused for as many calls to ``compute()`` as possible.

The GPU code currently assumes double precision and 2 dimensional coordinates for the point sets.

```
    #include "expdist.h"

    //instantiate GPUExpDist object
    GPUExpDist gpu_expdist(max_n);

    //call the cost function
    cost = gpu_expdist.compute(A, B, m, n, scale_A, scale_B);

```

The ``compute()`` function will take care of transferring the data to the GPU, calling 
the GPU functions, as well as
copying the results back from the GPU to host memory. It will only copy the actual size of
A and B, so there is no real performance drawback from overestimating the largest possible
size when creating GPUExpDist.
 
The destructor of GPUExpDist guarantees that GPU resources are freed when
the computations on the GPU have finished and the GPUExpDist instance is deleted.


## Installation

### CUDA

The GPU code requires a CUDA-capable GPU as well as the CUDA toolkit to be installed. Please see
Nvidia's website for installation fo the CUDA toolkit (https://developer.nvidia.com/cuda-downloads).

### CUB Library

The GPU code currently has one dependency, which is the CUB library. You can download it from:
https://nvlabs.github.io/cub/index.html
The easiest way to install CUB is to add the directory where you unpack CUB to your
``$CPATH`` environment variable.

### Python 3

The tests for the GPU code and several of the C functions are written in Python, to run these a
Python 3 installation is required. The easiest way to get this is using
[Miniconda](https://conda.io/miniconda.html).

On Linux systems one could type the following commands to download and install Python 3 using Miniconda:
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

All the required Python packages can be installed using the following command, before you run it
make sure that you have CUDA installed:
```
pip install -r requirements.txt
```

The tests can be run using ``nose``, for example by typing the following in the top-level or test
directory:
```
nosetests -v
```

### Building the code

A Makefile is provided that can be used to produce a dynamic library.
Simply type ``make`` in the top-level directory and the static library will be produced in the `bin/` directory.

Note that, the CUDA path inside the Makefile should be changed to match your local configuration.

The dynamic library can be used to link the code to your application, don't forget to also link the CUDA runtime library.
This means adding ``-L/path/to/cuda/lib64/ -lcudart`` to the compiler command when linking the application.

