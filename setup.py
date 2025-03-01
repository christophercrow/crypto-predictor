from setuptools import setup, find_packages
import os

# Read the contents of your README file for the long description.
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="crypto_predictor",
    version="0.1.0",
    description="A machine learning project for forecasting cryptocurrency prices using LSTM models with SHAP-based interpretability.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Christopher Crow",
    author_email="your-email@example.com",
    url="https://github.com/christophercrow/crypto-predictor",
    packages=find_packages(exclude=["tests*", "docs"]),
    install_requires=[
        "pandas>=1.0.0",
        "pyyaml>=5.1",
        "tensorflow>=2.0.0",
        "streamlit>=0.80.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "flake8>=3.8.0",
            "black>=22.0",
            "mypy>=0.800",
        ]
    },
    entry_points={
        "console_scripts": [
            "crypto_predictor=crypto_predictor.main:run_pipeline",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
