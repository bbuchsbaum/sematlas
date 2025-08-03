# Generative Brain Atlas - Makefile
# Usage: make <target>

.PHONY: help install download-neurosynth train test lint clean

help:
	@echo "Available targets:"
	@echo "  install            Create conda environment and install dependencies"
	@echo "  download-neurosynth Download Neurosynth database"
	@echo "  train              Train baseline VAE model"
	@echo "  test               Run test suite"
	@echo "  lint               Run code formatting and linting"
	@echo "  clean              Remove temporary files and caches"

install:
	conda env create -f environment.yml
	conda activate sematlas
	@echo "Environment created. Activate with: conda activate sematlas"

download-neurosynth:
	python scripts/download_neurosynth.py

train:
	python train.py

test:
	pytest tests/

lint:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/
	flake8 src/ scripts/ tests/

clean:
	rm -rf data/processed/*
	rm -rf .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete