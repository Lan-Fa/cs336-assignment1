from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "cs336_basics._cpp_tokenizer",
        sources=[
            "cs336_basics/cpp_tokenizer.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            "cs336_basics",
        ],
        language="c++",
        extra_compile_args=[
            "-O2",
            "-std=c++17",
        ],
    )
]

setup(
    name="cs336_basics_cpp_tokenizer",
    version="0.1.0",
    packages=["cs336_basics"],
    ext_modules=ext_modules,
)