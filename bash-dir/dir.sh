#!/bin/bash
# This script creates the directory structure for the crypto-predictor project

# Define the project root directory
PROJECT_DIR="crypto-predictor"

# Create the main project directories
mkdir -p "$PROJECT_DIR/config"          # Configuration files (YAML, JSON, etc.)
mkdir -p "$PROJECT_DIR/data"            # Raw and processed datasets
mkdir -p "$PROJECT_DIR/docker"          # Docker-related files
mkdir -p "$PROJECT_DIR/notebooks"       # Jupyter notebooks for exploration
mkdir -p "$PROJECT_DIR/src"             # Main source code
mkdir -p "$PROJECT_DIR/tests"           # Unit and integration tests
mkdir -p "$PROJECT_DIR/.github/workflows"  # GitHub Actions workflows for CI/CD

# Create the core files in the project root
touch "$PROJECT_DIR/.gitignore"         # Files and folders to ignore in Git
touch "$PROJECT_DIR/Dockerfile"         # Containerization configuration
touch "$PROJECT_DIR/requirements.txt"   # Pinned Python dependencies
touch "$PROJECT_DIR/README.md"          # Project overview and instructions

# Create the source code files in the src directory
touch "$PROJECT_DIR/src/__init__.py"        # Package initializer
touch "$PROJECT_DIR/src/data_ingestion.py"    # Module for data collection from APIs
touch "$PROJECT_DIR/src/preprocessing.py"     # Data cleaning and feature engineering
touch "$PROJECT_DIR/src/model.py"             # Deep learning model(s)
touch "$PROJECT_DIR/src/explainability.py"      # SHAP/LIME integration
touch "$PROJECT_DIR/src/utils.py"             # Utility functions (logging, config loading, etc.)
touch "$PROJECT_DIR/src/main.py"              # Entry point for training/inference pipelines

# Create the CI configuration file for GitHub Actions
touch "$PROJECT_DIR/.github/workflows/ci.yml"  # Continuous integration configuration

echo "Project structure for '$PROJECT_DIR' has been created successfully."
