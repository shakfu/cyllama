from setuptools import Extension, setup
from Cython.Build import cythonize

setup(
    name="callback",
    ext_modules=cythonize(
        [
            Extension("cheese", ["cheese.pyx", "cheesefinder.c"]),
        ]
    ),
)
