 # Loan Prediction Package

## Overview

Welcome to the Loan Prediction Package! This Python package is designed to help predict whether a loan will be approved based on a variety of financial features. It offers an end-to-end pipeline for loading, preprocessing, visualizing, training, and evaluating a machine learning model. Whether you're working with loan data in a personal project or as part of a larger business analysis, this package provides all the tools you need.

## Key Features

**Data Loading**: Easily load your loan dataset with a simple function.
**Data Preprocessing**: Clean and prepare your dataset for model training, including handling missing values and unnecessary columns.
**Visualization**: Create informative plots and graphs to explore the dataset and evaluate model performance.
**Model Training**: Train a Random Forest Classifier to predict loan approval based on the data.
**Model Evaluation**: Assess your model's performance with metrics like AUC score and Mean Squared Error.

## Installation

To get started, clone this repository and install the package locally:

```bash
git clone https://github.com/Pratiksha0221/Loan_prediction_package.git
cd loan_prediction
pip install -r requirements.txt
```

## Usage

Here's a quick guide to get you started using the package in your own analysis.

### 1. Importing the Package

To begin, you need to import the package and its modules into your script or Jupyter notebook:

```python
from loan_prediction import load_data, DataProcessor, ModelTrainer, Visualizer
```

### 2. Loading the Data

The first step is to load your loan dataset from a CSV file. The `load_data` function does just that:

```python
# Load data from CSV file
data = load_data('loan_data.csv')
```

Make sure your CSV file is structured with columns like `Age`, `Income`, `Education`, and `Personal.Loan`, as these are the features we use for prediction.

### 3. Preprocessing the Data

The `DataProcessor` class helps you clean and prepare the data for training by removing unnecessary columns and handling missing values.

```python
# Initialize the DataProcessor
processor = DataProcessor(data)

# Perform preprocessing
processed_data = processor.preprocess()
```

### 4. Visualizing the Data

To explore the dataset visually, use the `Visualizer` class. It will generate a variety of plots to help you understand the distribution of features:

```python
# Initialize Visualizer
visualizer = Visualizer(processed_data)

# Generate histograms for each feature
visualizer.visualize_data()
```

### 5. Training the Model

Once the data is ready, you can train a Random Forest Classifier using the `ModelTrainer` class. It provides methods to split the data, train the model, and evaluate its performance:

```python
# Initialize the ModelTrainer
trainer = ModelTrainer(processed_data)

# Split the data into training and test sets
trainer.split_data()

# Train the model
trainer.train_model()

# Evaluate the model
results = trainer.evaluate_model()
```

### 6. Model Evaluation

The `evaluate_model` method will output two important performance metrics:

- **AUC Score**: Measures the model's ability to distinguish between loan approvals and denials. A higher AUC score means better model performance.
- **Mean Squared Error**: A lower value indicates the model's predictions are close to the actual outcomes.

```python
print(f"AUC Score: {results['AUC Score']}")
print(f"Mean Squared Error: {results['Mean Squared Error']}")
```

### 7. Plotting the Confusion Matrix

To visualize how well the model performs, you can plot the confusion matrix:

```python
trainer.plot_confusion_matrix(trainer.y_test, trainer.model.predict(trainer.x_test))
```

## Conclusion

With this package, you can easily build and evaluate a loan prediction model. It provides essential steps for preparing data, training a machine learning model, and visualizing the results in a user-friendly way. 

## Contributing

We welcome contributions to this project. If you'd like to contribute, feel free to open an issue or submit a pull request with your changes. Please ensure that all code follows the project's coding standards.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

