# Comparative Analysis of Machine Learning Models for Bike Rental Prediction

## Project Overview

This project aims to predict bike rental demand by comparing various machine learning models. Accurate prediction of bike rentals is crucial for efficient fleet management and enhancing customer satisfaction. This project involves data preprocessing, model training, hyperparameter tuning, and performance evaluation using different regression algorithms.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Models](#models)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Technologies Used](#technologies-used)
- [Acknowledgments](#acknowledgments)

## Introduction

Bike rent methods are increasingly popular in urban areas, offering a convenient and eco-friendly mode of transportation. Predicting bike rental demand accurately aids in optimized fleet management and improves customer experience. This project compares multiple machine learning models to determine the best approach for predicting bike rentals.

## Dataset

The dataset used in this project has information about bike rentals, including various features like temperature, season, humidity, wind speed, and whether the day is a holiday or working day. The data was preprocessed to handle outliers and missing values.

## Data Preprocessing

1. **Handling Outliers:** Used Isolation Forest to detect and handle outliers in the wind speed data.
2. **Missing Values:** Replaced outlier values with NaN and handled missing data appropriately.
3. **Feature Selection:** Selected relevant features for the prediction task.
4. **Data Visualization:** Used Seaborn to visualize relationships between features and rental counts.

## Models

The following machine learning models were implemented and compared:
- **Decision Tree**
- **Random Forest**
- **Linear Regression**
- **SVC**
- **XGBoost**
- **AdaBoost**
- **GradientBoosting**
- **Stacking Regressor** (combination of Linear Regression and XGBoost)

## Model Evaluation

The models were evaluated based on the following metrics:
- **Mean Absolute Percentage Error (MAPE)**
- **R-squared (R2)**
- **Adjusted R-squared**

## Results

The Stacking Regressor, combining Linear Regression and XGBoost, outperformed the other models with the following results:
- **Mean Absolute Percentage Error:** 17.07%
- **R-squared:** 0.885
- **Adjusted R-squared:** 0.884

## Conclusion

The Stacking Regressor provided the best performance, indicating that combining multiple models can lead to better prediction accuracy. This approach can significantly enhance bike rental demand forecasting, aiding in more efficient fleet management and improved customer satisfaction.

## Technologies Used

- **Python**
- **Scikit-Learn**
- **XGBoost**
- **Matplotlib**
- **Seaborn**
- **Pandas**
- **Numpy**

## Acknowledgments

- Special thanks to the open-source community for providing the tools and libraries used in this project.
- Thanks to the contributors and maintainers of the dataset.
