import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;9.0'

setup(
    name='minference',
    ext_modules=[
        CUDAExtension(
            name='minference',
            sources=['./sparse_frontier/modelling/attention/minference/csrc/kernels.cpp', './sparse_frontier/modelling/attention/minference/csrc/vertical_slash_index.cu'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    package_dir={'': 'sparse_frontier/modelling/attention/minference'}
)
