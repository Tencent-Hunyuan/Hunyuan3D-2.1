import os
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import numpy

this_dir = os.path.dirname(__file__)
source_file = os.path.join(this_dir, "mesh_inpaint_processor.cpp")

ext = Pybind11Extension(
    "hy3dpaint.DifferentiableRenderer.mesh_inpaint_processor",
    [source_file],
    include_dirs=[pybind11.get_include(), numpy.get_include()],
    language="c++",
    extra_compile_args=["-O3", "-std=c++17"],
)

setup(
    name="mesh_inpaint_processor_build",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
)