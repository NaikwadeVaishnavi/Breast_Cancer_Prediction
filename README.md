# Breast Cancer Prediction

This repository contains code for a Breast Cancer Prediction project implemented in Python. The project uses the logistic regression algorithm to predict whether a breast cancer tumor is benign or malignant based on various features.

## Problem Statement: <br>

Breast cancer is one of the most common types of cancer among women globally. Early detection and accurate diagnosis are crucial for effective treatment and improved survival rates. This project aims to create a predictive model using machine learning to assist in the classification of breast tumors as either benign or malignant based on various features extracted from digitized images.

## Business Understanding<br>
The primary goal of this project is to develop a reliable and accurate predictive model that can assist medical professionals in diagnosing breast cancer. The model will take input features related to tumor characteristics and provide a binary classification, helping healthcare providers make informed decisions about patient treatment plans.


## Breast Cancer Prediction Algorithm

### Logistic Regression

Logistic regression is a widely used classification algorithm that models the probability of a binary outcome. In the context of this project, logistic regression is employed to predict whether a breast tumor is benign or malignant based on features extracted from breast cancer data.

The logistic regression algorithm works by fitting a sigmoid function to the input features, which maps the input data to a probability score between 0 and 1. If the probability score exceeds a certain threshold (typically 0.5), the model predicts the positive class (benign tumor); otherwise, it predicts the negative class (malignant tumor).



## Dataset

### Breast Cancer Dataset

The breast cancer dataset used in this project is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)). It consists of features extracted from digitized images of breast cancer tumors and corresponding labels indicating whether the tumor is benign or malignant.

The features include various measurements and characteristics of cell nuclei extracted from the images, such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

The target variable (label) indicates the diagnosis of the tumor, with '0' representing malignant tumors and '1' representing benign tumors.


Happy predicting!
