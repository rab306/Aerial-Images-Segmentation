# Satellite Images Semantic Segmentation

This repository contains code for a machine learning project that performs semantic segmentation on aerial imagery using U-Net architecture. The project aims to accurately segment different classes in aerial images.

## Project Overview

The project involves:

- Preprocessing and analyzing aerial imagery data.
- Implementing U-Net architecture for semantic segmentation.
- Training and evaluating the model on various aerial images.
- Visualizing segmentation results and model performance.

## Repository Structure

- `main.py`: Python script containing the main model training and evaluation code.
- `project_overview.ipynb`: Jupyter notebook that provides an overview of the project, data exploration, and visualizations.
- `requirements.txt`: File listing the required libraries for the project.
- `Src/`: Directory containing source code files, including model definitions and utility functions.
- `plots/`: Directory for storing generated plots and visualizations.

## Dataset

The dataset is available on Kaggle: [Semantic Segmentation of Aerial Imagery](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery).

### Dataset Description

The dataset consists of various aerial images that require semantic segmentation into different classes. Each image is labeled for training purposes to enable accurate segmentation by the model.

## Installation

To run the code, ensure you have Python 3 installed along with the required libraries listed in `requirements.txt`. You can install the dependencies using pip:

```bash
pip install -r requirements.txt
