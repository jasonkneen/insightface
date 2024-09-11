from setuptools import setup, find_packages

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
)
