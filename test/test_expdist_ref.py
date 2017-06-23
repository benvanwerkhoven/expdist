import numpy
from kernel_tuner import run_kernel

from test_utils import get_kernel_path, generate_inputs, call_reference_function


def test_expdist_ref():

    size = numpy.int32(100)
    ndim = numpy.int32(2)
    cost, A, B, scale_A, scale_B = generate_inputs(size, ndim, 1)

    arguments = [cost, A, B, size, size, ndim, scale_A, scale_B]

    with open(get_kernel_path()+'expdist_c.cpp', 'r') as f:
        kernel_string = f.read()

    answer = run_kernel("call_expdist", kernel_string, size, arguments, {},
               lang="C", compiler_options=['-I'+get_kernel_path()])

    cost = call_reference_function(size, ndim, A, B, scale_A, scale_B, cost)

    print("cost")
    print(cost)

    print("A")
    print(A)
    print("B")
    print(B)
    print("scale_A")
    print(scale_A)
    print("scale_B")
    print(scale_B)

    assert 100.0 < cost and cost < 200.0
