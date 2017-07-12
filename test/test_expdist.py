#!/usr/bin/env python
from nose.tools import nottest
import numpy
from kernel_tuner import run_kernel

from test_utils import get_kernel_path, generate_inputs, call_reference_function

compiler_options = ['-I'+get_kernel_path()]


def test_expdist_kernel():

    #setup test input
    allocation_size = int(3000)
    ndim = numpy.int32(2)
    size = numpy.int32(2000)

    params = dict()
    params["block_size_x"] = 32
    params["block_size_y"] = 4
    params["tile_size_x"] = 2
    params["tile_size_y"] = 4
    params["use_shared_mem"] = 1

    nblocks = numpy.int32( numpy.ceil(size / float(params["block_size_x"]*params["tile_size_x"])) *
                           numpy.ceil(size / float(params["block_size_y"]*params["tile_size_y"])) )
    print("nblocks")
    print(nblocks)

    cost, A, B, scale_A, scale_B = generate_inputs(allocation_size, ndim, nblocks)

    #call the reference function
    ref_cost = call_reference_function(size, ndim, A, B, scale_A, scale_B, cost)

    #call the GPU function
    with open(get_kernel_path()+'kernels.cu', 'r') as f:
        kernel_string = f.read()

    arguments = [A, B, size, size, scale_A, scale_B, cost]

    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]

    answer = run_kernel("ExpDist", kernel_string, (size, size), arguments, params,
               compiler_options=compiler_options, grid_div_x=grid_div_x, grid_div_y=grid_div_y)

    #collect the results from the first kernel
    cross_term = answer[6]
    print("intermediate cross_term")
    print(cross_term)

    #call the second kernel to reduce the per thread block cross terms to a single value
    out = numpy.zeros(1).astype(numpy.float64)

    arguments = [out, cross_term, size, size, nblocks]
    answer = run_kernel("reduce_cross_term", kernel_string, 1, arguments, {"block_size_x": 128},
               compiler_options=compiler_options, grid_div_x=[])

    #final cross term
    cost = answer[0][0]

    print("reference")
    print(ref_cost)
    print("answer")
    print(cost)

    print("reference")
    print("%30.20e" % ref_cost)
    print("answer")
    print("%30.20e" % cost)

    assert numpy.isclose(ref_cost, cost, atol=1e-5)



def test_expdist_kernel_column():

    #setup test input
    allocation_size = int(3000)
    ndim = numpy.int32(2)
    size = numpy.int32(2000)

    params = dict()
    params["block_size_x"] = 32
    params["block_size_y"] = 4
    params["tile_size_x"] = 2
    params["tile_size_y"] = 4
    params["use_shared_mem"] = 1

    nblocks = numpy.int32( numpy.ceil(size / float(params["block_size_x"]*params["tile_size_x"])) )

    print("nblocks")
    print(nblocks)

    cost, A, B, scale_A, scale_B = generate_inputs(allocation_size, ndim, nblocks)

    #call the reference function
    ref_cost = call_reference_function(size, ndim, A, B, scale_A, scale_B, cost)

    #call the GPU function
    with open(get_kernel_path()+'kernels.cu', 'r') as f:
        kernel_string = f.read()

    arguments = [A, B, size, size, scale_A, scale_B, cost]

    grid_div_x = ["block_size_x", "tile_size_x"]

    answer = run_kernel("ExpDist_column", kernel_string, size, arguments, params,
               compiler_options=compiler_options, grid_div_x=grid_div_x)

    #collect the results from the first kernel
    cross_term = answer[6]
    print("intermediate cross_term")
    print(cross_term)

    #call the second kernel to reduce the per thread block cross terms to a single value
    out = numpy.zeros(1).astype(numpy.float64)

    arguments = [out, cross_term, size, size, nblocks]
    answer = run_kernel("reduce_cross_term", kernel_string, 1, arguments, {"block_size_x": 128},
               compiler_options=compiler_options, grid_div_x=[])

    #final cross term
    cost = answer[0][0]

    print("reference")
    print(ref_cost)
    print("answer")
    print(cost)

    print("reference")
    print("%30.20e" % ref_cost)
    print("answer")
    print("%30.20e" % cost)

    assert numpy.isclose(ref_cost, cost, atol=1e-5)




def test_hostfunction():

    #setup test input
    allocation_size = numpy.int32(3000)
    size = numpy.int32(2000)
    ndim = numpy.int32(2)

    nblocks = numpy.int32(numpy.ceil(size / (32*4)) * numpy.ceil(size / (4*4)))

    cost, A, B, scale_A, scale_B = generate_inputs(allocation_size, ndim, nblocks)

    #call the reference function
    ref_cost = call_reference_function(size, ndim, A, B, scale_A, scale_B, cost)

    #call the host function
    arguments = [cost, A, B, size, size, ndim, scale_A, scale_B, allocation_size]
    with open(get_kernel_path()+'expdist.cu', 'r') as f:
        kernel_string = f.read()
    answer = run_kernel("test_GPUExpDistHost", kernel_string, size, arguments, {},
               lang="C", compiler_options=compiler_options+['-arch=sm_30'])
    cost = answer[0][0]

    print("reference")
    print(ref_cost)

    print("answer")
    print(cost)

    assert numpy.isclose(ref_cost, cost, atol=1e-5)





def test_hostfunction_largeN():

    #setup test input
    allocation_size = numpy.int32(1e6)
    size = numpy.int32(40000)
    ndim = numpy.int32(2)

    params = dict()
    params["block_size_x"] = 32
    params["block_size_y"] = 4
    params["tile_size_x"] = 2
    params["tile_size_y"] = 4
    params["use_shared_mem"] = 1

    #compute nblocks for when using the expdist kernel
    nblocks = numpy.int32( numpy.ceil(size / float(params["block_size_x"]*params["tile_size_x"])) *
                           numpy.ceil(size / float(params["block_size_y"]*params["tile_size_y"])) )

    #ensure that this test actually causes the host code to call the column kernel
    assert nblocks > allocation_size

    #compute the nblocks actually used by the column kernel
    nblocks = numpy.int32(numpy.ceil(size / float(params["block_size_x"] * params["tile_size_x"])))

    #generate input data
    cost, A, B, scale_A, scale_B = generate_inputs(allocation_size, ndim, nblocks)

    #call the ExpDist_column kernel directly for reference
    arguments = [A, B, size, size, scale_A, scale_B, cost]
    grid_div_x = ["block_size_x", "tile_size_x"]
    with open(get_kernel_path()+'kernels.cu', 'r') as f:
        kernel_string = f.read()
    answer = run_kernel("ExpDist_column", kernel_string, size, arguments, params,
               compiler_options=compiler_options, grid_div_x=grid_div_x)
    ref_cost = numpy.sum(answer[6])

    #call the host function
    arguments = [cost, A, B, size, size, ndim, scale_A, scale_B, allocation_size]
    with open(get_kernel_path()+'expdist.cu', 'r') as f:
        kernel_string = f.read()
    answer = run_kernel("test_GPUExpDistHost", kernel_string, size, arguments, {},
               lang="C", compiler_options=compiler_options+['-arch=sm_30'])
    cost = answer[0][0]

    print("reference")
    print(ref_cost)

    print("answer")
    print(cost)

    assert numpy.isclose(ref_cost, cost, atol=1e-5)


if __name__ == "__main__":
    test_expdist_kernel()
