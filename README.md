<img src="./docs/assets/images/foto.jpg" width=165 height=165 align="right">

# NLP Project

***It is a project carried out by Pablo Sánchez Cabrera that aims to answer a series of questions
about the behavior of American politicians and senators on social networks.***

During this project, a prediction of the sentiment of the tweets as well as their theme has been made.
To do this, the Huggig Face platform has been used. 
Thus, two types of Transformers architectures designed specifically for both task will be considered.

In the case of topic prediction, a Zero-Shot architecture has been used; 
which is a type of algorithm that allows learning classes that have not previously been seen.
Below is a graphic representation of this type of architecture

<img src="./docs/assets/images/zero-shot.png" width=1000 height=400 align="center">

## Output

The results of the studies carried out can be seen in the `development` file in the `notebooks` folder.
## Structure

```
│
├── README.md                               # This file                             
│                              
├── src                                     # source code
|   └── majorel 
│       └── _utils                          # Utils
│       └── graphs                          # Visualization and plots    
│       └── preprocessing                   # Text Preprocessing                      
│       └── model
│           └── model_task                  # Sentiment Analysis model and Zero-Shot Class. (Topic Prediction)
│
├── test                                    # unit testing   
│    └── test_preprocessing                 # unit tests for preprocessing
│    └── test_model_task                    # unit tests for classification task (sentiment and topic)
│
├── data                                    # data folder with input files to be used in the tests
│
├── docs                                    # Documentation folder 
│    └── index.md                           # index of the documentation
│    └── assets                             # assets of the documentation
│    └── start                              # start screen of the documentation
│    └── user_guides                        # user guide notebooks
│    └── modules                            # API documentation
│
├── requirements.txt                        # requirements list
│
├── dev-requirements.txt                    # requirements list for building the documentation
│
│── mkdocs.yml                              # mkdocs config file for building the documentation
```

## How to set up the environment

Create conda environment
```
conda create --name majorel python=3.9
```
Activate conda environment
```
conda activate majorel 
```

## Dependencies

```
aiohttp==3.7.4
pandas==1.2.2
xlrd==2.0.1
numpy==1.21
beartype==0.9.0
openpyxl==3.0.10
scipy==1.8.1
scikit-learn==1.1.1
scikit-optimize==0.9.0
hyperopt==0.2.7
stumpy==1.11.1
h5py==3.6.0
tune_sklearn==0.4.1
tensorflow==2.9.0
matplotlib==3.5.2
plotly==5.8.0
plotly-express==0.4.1
nbformat==5.4.0
kaleido==0.2.1
protobuf==3.20.0
xgboost==1.5.0
lightgbm==3.2.1
ipywidgets==7.6.3
shap==0.40.0
fronni==0.0.6
spacy==3.4
sentence_transformers==2.2.2
ipykernel==6.15.2
jupyterlab==3.4.6
```

# Features

Throughout this project, different tasks have been carried out:

### Graphs

Barplot and histogram

### Preprocessing

Removing characters (@, RT, #, etc.) including suppression of stop-words or lemmatization of the text.

### Machine Learning model

Development of a system for predicting the theme and sentiment of the comments 
made by different politicians on social networks.

# Documentation

Command to display the documentation on a local server:

```
mkdocs serve
```

## Dependencies

```
flake8
mkdocs==1.3.0
mkdocs-autorefs==0.4.1
mkdocs-gen-files==0.3.4
mkdocs-jupyter==0.21.0
mkdocs-literate-nav==0.4.1
mkdocs-material==8.3.0
mkdocs-material-extensions==1.0.3
mkdocs-pdf-export-plugin==0.5.10
mkdocstrings-python==0.7.0
```

## Other Considerations

Code developed according to the quality standards for Software Development.
- PEP 8: Style Guide for Python Code
- Modular code
- Development of unit tests

The project folder structure follows the Cookiecutter tool: 
https://drivendata.github.io/cookiecutter-data-science/

## License

(C) Pablo Sánchez Cabrera - 2022