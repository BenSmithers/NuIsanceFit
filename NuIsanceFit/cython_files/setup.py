from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension    
from Cython.Distutils import build_ext

ext_modules = [Extension("event", sources=["event.pyx"],
            ),
            Extension("weighter",
                sources=["weighter.pyx"],
                libraries=["photospline"],
                language="c++",
                extra_compile_args=["-fopenmp", "-O3", "-I/usr/local/include/photospline/"],
                extra_link_args=["-DSOME_DEFINE_OPT","-L/usr/local/lib/"]
            )]

setup(
    name="NuIsanceFit",
    cmdclass={'build_ext':build_ext},
    compiler_directives={'language_level':"3"},
    ext_modules=ext_modules
)
