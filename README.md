
# Anomaly Detection in CREDO Dataset

The goal of the project is to perform anomaly detection on images using different techniques.


## Description

This project focuses on analyzing anomalies in the CREADO dataset, which consists of images captured by a MOS sensor capturing cosmic radiation particles. The primary objective is to identify and understand unusual patterns or outliers within the dataset. 

To achieve this, we employed Python and implemented various anomaly detection techniques, including:
- Principal Component Analysis (PCA) 
- Autoencoders 
- 2D PCA

The CREADO Anomaly Analysis Project provides a comprehensive exploration of anomaly detection in the context of cosmic radiation imagery. By leveraging PCA, Autoencoders, and 2D PCA, we aim to contribute valuable insights into identifying and understanding anomalies within the CREADO dataset. 
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


## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

## Acknowledgements

 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)


