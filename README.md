# Fraud Detection System

## Overview

This repository contains a Jupyter Notebook (`fraud_detection.ipynb`) that implements a comprehensive fraud detection system using machine learning. The system is designed to analyze transaction data and identify potentially fraudulent activities. The notebook covers the entire data science pipeline, from data preprocessing to model evaluation.

## Features

- **Data Preprocessing**: Handles missing values, outliers, and encodes categorical variables.
- **Exploratory Data Analysis (EDA)**: Includes visualizations to understand data distributions and relationships.
- **Feature Engineering**: Creates new features to improve model performance.
- **Machine Learning Models**: Implements multiple models including Random Forest, Gradient Boosting, Logistic Regression, and more.
- **Model Evaluation**: Uses metrics like accuracy, precision, recall, F1-score, and ROC-AUC to evaluate performance.
- **Hyperparameter Tuning**: Optimizes model parameters using RandomizedSearchCV.
- **Handling Class Imbalance**: Uses SMOTE to address imbalanced datasets.

## Dataset

The dataset used in this project (`student_dataset.csv`) contains transaction data with the following features:
- Transaction.Date
- Transaction.Amount
- Customer.Age
- Account.Age.Days
- Transaction.Hour
- source
- browser
- Payment.Method
- Product.Category
- Quantity
- Device.Used
- Is.Fraudulent (target variable)

## Dependencies

To run this notebook, you'll need the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn (imblearn)
- category_encoders
- xgboost
- lightgbm

You can install the required libraries using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn category_encoders xgboost lightgbm
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
```

2. Open the Jupyter Notebook:
```bash
jupyter notebook fraud_detection.ipynb
```

3. Follow the steps in the notebook to preprocess the data, train models, and evaluate performance.

## Results

The notebook demonstrates the performance of various machine learning models on the fraud detection task. Key metrics are reported for each model, allowing for easy comparison of their effectiveness.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License

This project is open-source and available under the [MIT License](LICENSE).

## Contact

For questions or suggestions, please open an issue in the repository or contact the maintainer directly.
