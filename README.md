# House-Price-Prediction

This project focuses on predicting house prices using machine learning techniques. Based on various features like square footage, number of bedrooms and bathrooms, house age, and location quality, the model aims to accurately estimate house prices. The project is intended to demonstrate data preprocessing, feature engineering, and the application of regression algorithms for price prediction.

**Project Overview**

House price prediction is a common use case in the real estate industry, where accurate predictions can help buyers, sellers, and investors make informed decisions. 
In this project:

A dataset with features such as square footage, number of bedrooms, number of bathrooms, house age, and location quality is used.
The data is preprocessed to handle missing values and scale numerical features.
Machine learning algorithms, including Linear Regression and Decision Tree Regression, are implemented and evaluated based on their Mean Squared Error (MSE) on test data.
The model with the lowest MSE is recommended for making predictions.

**Features**

The dataset includes the following features:
  square_footage: The total square footage of the house.
  num_bedrooms: Number of bedrooms in the house.
  num_bathrooms: Number of bathrooms in the house.
  age_of_house: Age of the house in years.
  location_quality: A score from 1 to 5 indicating the desirability of the house's location (1 being the lowest and 5 the highest).
  price: The target variable, representing the actual price of the house.
 
**Prerequisites**

  Python 3.x
  Libraries: pandas, numpy, scikit-learn
