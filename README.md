# Crypto Predictor

Crypto Predictor is an end-to-end machine learning project for forecasting cryptocurrency prices using LSTM models with SHAP-based interpretability. It is built as an installable package for easier maintenance and extensibility.

![Build Status](https://github.com/christophercrow/crypto-predictor/workflows/CI/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Overview
This project leverages deep learning techniques to predict cryptocurrency prices. It includes modules for data ingestion, preprocessing, model training, and an interactive Streamlit dashboard for interpretability.

## Getting Started

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/christophercrow/crypto-predictor.git
cd crypto-predictor
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -e .
