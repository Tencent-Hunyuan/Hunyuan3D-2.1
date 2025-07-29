import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

here = os.path.abspath(os.path.dirname(__file__))

ext_modules = [
    CppExtension(
        'mesh_inpaint_processor',
        ['mesh_inpaint_processor.cpp'],
        extra_compile_args={'cxx': ['/std:c++17']} if os.name == 'nt' else ['-std=c++17']
    )
]

setup(
    name='mesh_inpaint_processor',
    version='0.1.0',
    author='Hunyuan3D Team (with Windows build by Adilet)',
    description='A custom C++ processor for mesh inpainting.',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False,
)