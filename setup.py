from setuptools import setup, find_packages

setup(
    name="crypto_predictor",
    version="0.1.0",
    packages=find_packages(),  # Finds the crypto_predictor package
    install_requires=[
        "tensorflow>=2.10",
        "pandas",
        "numpy",
        "scikit-learn",
        "pyyaml",
        "streamlit",
        "plotly",
        "shap",
        "requests",
    ],
    description="Cryptocurrency price prediction using LSTM and SHAP for interpretability.",
    author="Christopher Crow",
)
