import os
import numpy

from kernel_tuner import run_kernel
from nose.tools import nottest

@nottest
def get_kernel_path():
    """ get path to the kernels as a string """
    path = "/".join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
    return path+'/src/'

@nottest
def generate_inputs(size, ndim=2, nblocks=1):
    A = numpy.random.randn(size*ndim).astype(numpy.float64)
    B = A+0.00001*numpy.random.randn(size*ndim).astype(numpy.float64)
    scale_A = numpy.absolute(0.01*numpy.random.randn(size).astype(numpy.float64))
    scale_B = numpy.absolute(0.01*numpy.random.randn(size).astype(numpy.float64))
    cost = numpy.zeros((nblocks)).astype(numpy.float64)
    return cost, A, B, scale_A, scale_B

@nottest
def call_reference_function(size, ndim, A, B, scale_A, scale_B, cost):
    arguments = [cost, A, B, size, size, ndim, scale_A, scale_B]
    with open(get_kernel_path()+'expdist_c.cpp', 'r') as f:
        kernel_string = f.read()
    answer = run_kernel("call_expdist", kernel_string, size, arguments, {},
               lang="C", compiler_options=['-I'+get_kernel_path()])
    ref_cost = answer[0][0]
    return ref_cost


