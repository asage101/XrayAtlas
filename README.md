
# XrayAtlas: Automated X-Ray Classification and Registration

XrayAtlas is a computer vision and machine learning application developed to automate X-ray image classification and registration.

The project explores how image processing and deep learning techniques can be integrated into medical imaging workflows to support image organization, analysis, and alignment.

**Course:** CMPT 419 – Biomedical Image Computing  
**Institution:** Simon Fraser University  
**Date:** Spring 2025  
**Project Type:** Undergraduate group project

## Project Overview

Medical imaging workflows often involve large collections of images that require identification, classification, and alignment.

XrayAtlas was developed to investigate two components of this process:

1. **Image Classification:** Using machine learning to identify anatomical regions represented in X-ray images.
2. **Image Registration:** Aligning medical images to support comparison and further analysis.

The project combines these components into a Python-based application.

## Project Demo

[![Watch XrayAtlas Demo](https://img.shields.io/badge/Watch_Demo-YouTube-red?logo=youtube)](https://youtu.be/61XbSw7Ck1A)

[![XrayAtlas Demo](./images/XrayAtlas.png)](https://youtu.be/61XbSw7Ck1A)

## Technical Methods

### X-Ray Image Classification

The classification component uses convolutional neural networks (CNNs) to predict anatomical regions from X-ray images.

Techniques explored include:

- Image preprocessing and normalization
- Data augmentation
- Model training using TensorFlow/Keras
- Addressing imbalanced training data
- Evaluating classification performance

### Image Registration

The registration component is intended to align X-ray images to facilitate comparison and analysis.

The application combines image classification and registration within a unified workflow.

## Technologies

- **Programming:** Python
- **Machine Learning:** TensorFlow / Keras
- **Image Processing:** Computer vision and image registration
- **Development:** Jupyter Notebook, Conda
- **Dataset:** UNIFESP X-ray Body Part Classifier dataset

## Repository Structure

```text
XrayAtlas/
├── src/                    # Source code
├── Classification.ipynb   # Classification notebook
├── images/                 # Project screenshots
├── predict_image_here/     # Sample image input directory
├── main.py                 # Application entry point
├── requirements.yml        # Conda environment
└── README.md               # Project documentation
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/XrayAtlas.git
cd XrayAtlas
```

### 2. Create the Conda environment

```bash
conda env create -f requirements.yml
conda activate xray_atlas
```

### 3. Download the dataset

The classification component uses the
[UNIFESP X-ray Body Part Classifier dataset](https://www.kaggle.com/competitions/unifesp-x-ray-body-part-classifier/data).

Install and configure the Kaggle API, accept the dataset's access requirements, and download the data:

```bash
mkdir data
kaggle competitions download -c unifesp-x-ray-body-part-classifier -p data
```

Extract the downloaded archive into the directory expected by the application.

## Running the Application

Place a supported sample X-ray image in the `predict_image_here` directory.

Run:

```bash
python main.py --input ./predict_image_here/sample.png
```

The application processes the input image through the implemented classification and registration workflow.

## Academic Context

This project was developed collaboratively as part of CMPT 419: Biomedical Image Computing at Simon Fraser University.

The project provided practical experience with medical image analysis, machine learning, computer vision, and integrating computational methods into an application.

**Contributors:** Mykhailo Isyp, Niall Shaidullin, Eric Leeson, Ziqian Fan

## Further Information

For an overview of the application and its functionality, see the project demonstration linked above.
