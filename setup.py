# example file to compile a cython module
#
# python setup.py build_ext --inplace
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("_convolve", ["_convolve.pyx"])],
    include_dirs = [numpy.get_include(),],
)
