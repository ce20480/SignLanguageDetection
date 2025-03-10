from setuptools import setup, find_packages

setup(
    name="asl-detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "torch>=1.9.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0",
        "mediapipe>=0.8.9",
        "pyyaml>=5.4.0"
    ],
    author="ASL Detection Team",
    author_email="avinibusiness@gmail.com",
    description="A modular framework for ASL sign language detection",
    keywords="asl, sign language, computer vision, deep learning",
    python_requires=">=3.7",
)
