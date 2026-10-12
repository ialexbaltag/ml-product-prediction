# ml-product-prediction

Category prediction model - ML Project

In this repository is code for a machine learning pipeline for product category prediction based 
on products titles and their brands. I used script in Python with libraries scikit-learn.

This is demonstrative machine learning module where raw data used for trained model prediction.

Project contains: 

data/products.csv

notebook/ml_prediction_product.ipynb

src/test_model.py
src/train_model.py

README.md

Steps covered like a real-world ML project:

1.Project Setup
* Created a github repository
* Defined folder structure
* Uploaded raw dataset
* Create and syncronized notebook from Google Colab and localy

2.Data Exploration
* Loaded and analyzed dataset from csv file
* Used matplotlib and seaborn for visualizations
* Research for text characteristics

3.Cleaning the data and preprocessing
* Removed missing values
* Standardized category labels and the titles of the columns

4.Feature Engineering
* Used meaningful input features: product_title, product_title_length and product_brand
* Removed irrelevant columns

5.Model Training & Evaluation
* Compared multiple ML models (Logistic Regression, Naive Bayes, Decision Tree, Random Forest, SVM)
* Used ColumnTransformer and Pipeline for unified preprocessing
* Evaluated using precision, recall, F1-score and confusion matrix

6.Final Model Training
* Trained final model on full dataset
* Saved the pipeline using joblib to category_model.pkl

7.Inference & Usage
* Loaded saved model
* Built an interactive interface for predictiong category of product title
* Enable real-time testing via console input

How to use:

Train the model: cd src
                 python train_model.py

In the root directory will be created category_model.pkl file

Run Inferrence:
    Based on model_test.py predict the category of a product.

Author:
    This is a small project in educational purpose demonstrating how 
    to work with ML.

License:
    This is open-source project and freely available for educational use.



