# FIFA World Cup 2022 Tweets Sentiment Analysis

## Project Overview

This project implements a sentiment analysis system to classify tweets from the first day of the FIFA World Cup 2022 as positive, negative, or neutral. We developed and compared three different neural network architectures: Dense Neural Networks, vanilla Recurrent Neural Networks (RNN), and Long Short-Term Memory (LSTM) networks.

## Team Members

- Juan Diego Lora
- Sebastian Erazo
- Oscar Muñoz

## Dataset

The project uses a dataset containing tweets related to the FIFA World Cup 2022 with the following attributes:

- id
- Date Created
- Number of Likes
- Source of Tweet
- Tweet
- Sentiment

The dataset was cleaned and preprocessed in the `Preprocessing_FIFA_Tweets_NLTK.ipynb` notebook to ensure data quality and consistency across all attributes.

## Solution Implementation

### Data Preprocessing

- **Dataset Cleaning**: Comprehensive data cleaning process to ensure clean and consistent attributes
- Text preprocessing using NLTK (tokenization, lowercasing, stopword removal)
- Exploratory Data Analysis (EDA) to understand data distribution and patterns
- Data cleaning and preparation for supervised learning

### Models Implemented

#### 1. Baseline Model

- **DummyClassifier**: Implemented as a baseline to establish performance benchmarks
- Evaluated using accuracy, precision, recall, and F1-score metrics

#### 2. Dense Neural Network

- Feedforward neural network architecture for sentiment classification
- Hyperparameter optimization using GridSearchCV
- Performance evaluation with accuracy, precision, recall, F1-score, and kappa metrics

#### 3. Vanilla Recurrent Neural Network (RNN)

- Recurrent neural network implementation for sequence modeling
- GridSearchCV for hyperparameter tuning
- Comprehensive evaluation metrics

#### 4. Long Short-Term Memory (LSTM)

- LSTM architecture for capturing long-term dependencies in text
- Optimized hyperparameters through GridSearchCV
- Full performance evaluation including kappa coefficient

### Project Structure

```
├── data/                          # Raw dataset
├── data_processed/                # Preprocessed data
├── models/                        # Trained model files
├── notebooks/                     # Jupyter notebooks
│   ├── EDA.ipynb                 # Exploratory Data Analysis
│   ├── Preprocessing_FIFA_Tweets_NLTK.ipynb  # Data preprocessing
│   ├── Dummy_classifier.ipynb    # Baseline model
│   ├── Grid/                     # Models with GridSearch
│   │   ├── Dense_nn.ipynb
│   │   ├── Lstm_model.ipynb
│   │   └── VanillaRNNgrid.ipynb
│   └── WithoutGrid/              # Models without GridSearch
│       ├── Dense_nn_no_GridSearch.ipynb
│       ├── LSTM_sin_GridSearch.ipynb
│       └── SimpleRNNsinGrid.ipynb
├── docs/                         # Project documentation
└── requirements.txt              # Python dependencies
```

### Key Features

- **Comprehensive Model Comparison**: Three different neural network architectures
- **Hyperparameter Optimization**: GridSearchCV implementation for all models
- **Performance Metrics**: Accuracy, precision, recall, F1-score, and kappa evaluation
- **Data Visualization**: EDA with insights into tweet sentiment distribution
- **Modular Code Structure**: Well-organized notebooks for each model type

### Results

The project successfully implemented and compared multiple neural network architectures for sentiment analysis. Each model was evaluated using standard metrics, providing insights into the effectiveness of different approaches for tweet sentiment classification.

### Technologies Used

- Python
- TensorFlow/Keras
- NLTK for text preprocessing
- Scikit-learn for evaluation metrics
- Jupyter Notebooks for development
- Pandas and NumPy for data manipulation

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Navigate to the `notebooks/` directory
3. Start with `EDA.ipynb` for data exploration
4. Follow the preprocessing notebook for data preparation
5. Run individual model notebooks for sentiment classification

## Documentation

- Complete project report and presentation available in the `docs/` directory
- Detailed code comments and documentation in each notebook
- Model performance comparisons and analysis included
