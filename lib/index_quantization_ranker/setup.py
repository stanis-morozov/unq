# System imports
from distutils.core import *
from distutils      import sysconfig

# Third-party modules - we depend on numpy for everything
import numpy
import os

os.environ['CC'] = 'g++ -fpic -std=c++17 -O3 -march=native'

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# extension module
module = Extension("_index_quantization_ranker",
                   ["index_quantization_ranker.i","index_quantization_ranker.cpp"],
                   include_dirs=[numpy_include],
                   extra_compile_args=["-fopenmp"],
                   extra_link_args=['-lgomp'],
                   swig_opts=['-c++']
                   )

# setup
setup(  name        = "index_quantization_ranker",
        description = "Quick similarity search with FAISS quantization for custom distances",
        author      = "Yandex Research",
        version     = "1.0",
        ext_modules = [module]
)