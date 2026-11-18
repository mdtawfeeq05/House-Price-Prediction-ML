# House Price Prediction Using Machine Learning

## Overview

A beginner-friendly Machine Learning project that predicts house prices using Multiple Linear Regression. The project includes a Streamlit web interface for entering house details and generating predictions.

**Note:** This is an educational prototype trained on synthetic housing data. It is not intended for real-world property valuation.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Joblib

## Input Features

The model uses three features:

1. House area (square feet)
2. Number of bedrooms
3. House age (years)

The target variable is the house price in lakh rupees.

## Machine Learning Workflow

1. Generate a synthetic housing dataset.
2. Prepare input features (X) and target values (y).
3. Split the dataset into 80% training and 20% testing.
4. Train a Multiple Linear Regression model.
5. Predict prices on the test set.
6. Evaluate performance using MAE, MSE, RMSE, and R².
7. Save the trained model using Joblib.
8. Create a Streamlit interface for predictions.

## Model Evaluation

Results from the initial 20-record synthetic dataset:

| Metric | Result |
|---|---|
| MAE | ₹1.89 lakh |
| MSE | 5.07 (lakh rupees)² |
| RMSE | ₹2.25 lakh |
| R² Score | 0.998 |

These results are from a small synthetic dataset and do not establish real-world predictive accuracy.

## Example Prediction

**Input:**
- Area: 1,750 sq. ft.
- Bedrooms: 3
- Age: 4 years

**Predicted Price:** Approximately ₹74.81 lakh.

## How to Run

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Train and save the model:

```bash
python house_price_project.py
```

Launch the application:

```bash
python -m streamlit run app.py
```

## Future Improvements

- Train on a larger real-world housing dataset.
- Add location and property-type features.
- Compare Linear Regression with Random Forest.
- Apply cross-validation and hyperparameter tuning.
- Deploy the Streamlit application online.

## Learning Outcomes

This project demonstrates data preparation, supervised learning, regression modeling, model evaluation, model serialization, and basic ML application development.
