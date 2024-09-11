from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

# Define the Cython extension
extensions = [
    Extension(
        "insightface.thirdparty.face3d.mesh.cython.mesh_core_cython",
        ["insightface/thirdparty/face3d/mesh/cython/mesh_core_cython.pyx"],
        include_dirs=[np.get_include()]
    )
]

setup(
    name="insightface",
    version="0.7.3",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "onnx",
        "opencv-python",
        "tqdm",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "easydict",
        "matplotlib",
        "Pillow",
        "albumentations",
        "cython",
    ],
    python_requires='>=3.6',
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
)
