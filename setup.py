"""
DeepGuard-X: Production-Grade Multi-Modal Deepfake Detection System
Setup configuration for package installation
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="deepguard-x",
    version="1.0.0",
    author="DeepGuard-X Team",
    author_email="support@deepguard-x.ai",
    description="Production-grade multi-modal deepfake detection system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourorg/deepguard-x",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Sound/Audio",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.4",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
        ],
        "gpu": [
            "onnxruntime-gpu>=1.16.3",
        ],
        "full": [
            "foolbox>=3.3.3",
            "adversarial-robustness-toolbox>=1.17.0",
            "captum>=0.7.0",
            "shap>=0.44.0",
            "optuna>=3.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "deepguard-train-video=scripts.train_video:main",
            "deepguard-train-audio=scripts.train_audio:main",
            "deepguard-inference=scripts.inference:main",
            "deepguard-dashboard=src.dashboard.app:main",
            "deepguard-realtime=scripts.realtime_demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
