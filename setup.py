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
    ],
    author="Avini",
    author_email="avinibusiness@gmail.com",
    description="ASL detection using computer vision and deep learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/SignLanguageDetection",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires="3.9-3.12",
)
