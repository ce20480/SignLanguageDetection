from setuptools import find_packages, setup

setup(
    name="sl_detection",
    version="0.1.3",
    packages=find_packages(include=["sl_detection", "sl_detection.*"]),
    install_requires=[
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "torch>=1.9.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0",
        "mediapipe>=0.10.18",
    ],
    author="Avini",
    author_email="avinibusiness@gmail.com",
    description="SL detection using computer vision and deep learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ce20480/SignLanguageDetection",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9, <=3.12",
)
