from setuptools import setup
from Cython.Build import cythonize
import numpy

# used to compile the pyx files
# run: python setup.py build_ext --inplace
setup(
    ext_modules = cythonize(["nms.pyx","nms_cnn2.pyx"]),
    include_dirs=[numpy.get_include()]
)