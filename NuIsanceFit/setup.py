from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension    
from Cython.Distutils import build_ext

setup(
    ext_modules = cythonize(["event.pyx", "weighter.pyx"])
)

