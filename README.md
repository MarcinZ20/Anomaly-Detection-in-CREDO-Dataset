<!-- TOC --><a name="anomaly-detection-in-credo-dataset"></a>
# Anomaly Detection in CREDO Dataset

The goal of the project is to perform anomaly detection on images using different Machine Learning techniques.

## Table of contents
- [Anomaly Detection in CREDO Dataset](#anomaly-detection-in-credo-dataset)
   * [Description](#description)
   * [Tech-stack](#tech-stack)
   * [Project structure](#project-structure)
   * [Run Locally](#run-locally)
   * [Authors](#authors)
   * [References](#references)
   * [License](#license)
   * [Acknowledgements](#acknowledgements)

## Description

This project focuses on analyzing anomalies in the CREADO dataset. It consists of images registered by CMOS sensors scattered around the world, capturing cosmic radiation particles. The primary objective is to identify and understand unusual patterns, detect outliers within the dataset and test different approaches.

To achieve this, we employed Python and implemented various anomaly detection techniques, including:
- Principal Component Analysis (PCA) 
- Autoencoders 
- 2D PCA
- Morphological Methods

The CREADO Anomaly Analysis Project provides a comprehensive exploration of anomaly detection in the context of cosmic radiation imagery. By leveraging PCA, Autoencoders, and 2D PCA, we aim to contribute valuable insights into identifying and understanding anomalies within the CREADO dataset. 

## Tech-stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

## Project structure

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── plots              <- Plots extracted from notebooks
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


## Run Locally

Clone the project

```bash
  git clone https://github.com/MarcinZ20/Anomaly-Detection-in-CREDO-Dataset.git
```

Go to the project directory

```sh
cd Anomaly-Detection-in-CREDO-Dataset
```

Create environment
```sh
make create_environment
```

Install dependencies

```sh
make requirements
```

Verify installed environment

```sh
make test-environment
```

Create Dataset

```sh
make data
```

Images from `data/raw` should now be processed and loaded into `data/processed` directory


## Authors

- [@Marcin](https://www.github.com/MarcinZ20)
- [@Jan](https://www.github.com/tycjantyc)


## References
- PCA implementation algorithm used in the study: [PCA Implementation](https://github.com/Parveshdhull/FaceRecognitionUsing-PCA-2D-PCA-And-2D-Square-PCA)  
- Cookiecutter Data Science project template: [Cookiecutter](https://www.cookiecutter.io)


## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/) 
![GitHub Repo stars](https://img.shields.io/github/stars/MarcinZ20/Anomaly-Detection-In-Credo-Dataset?style=flat&logo=github)


## Acknowledgements

 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)


