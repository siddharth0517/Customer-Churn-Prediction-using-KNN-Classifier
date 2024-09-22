# Customer Churn Prediction Using KNN Classifier

## Overview
This project focuses on predicting **customer churn** using the **K-Nearest Neighbors (KNN) classification** algorithm. Customer churn refers to when customers stop doing business with a company. Predicting churn can help businesses retain customers by taking preventive measures based on the insights derived from the data.

## Project Objectives
+ Develop a machine learning model to predict whether a customer will churn.
+ Perform data preprocessing, including handling missing values, encoding categorical data, and feature scaling.
+ Use the K-Nearest Neighbors (KNN) algorithm to classify customers as likely to churn or not.
+ Evaluate the model performance using metrics such as accuracy, confusion matrix.

# Dataset
The dataset used for this project includes customer information such as demographics, services subscribed to, and account details. The target variable is whether a customer has churned (Yes) or not (No).
[link](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset) for Dataset

## Key Features:
+ **CustomerID:** Unique identifier for each customer.
+ **Gender:** Male or Female.
+ **SeniorCitizen:** Whether the customer is a senior citizen or not.
+ **Tenure:** Number of months the customer has been with the company.
+ **MonthlyCharges:** The monthly charges incurred by the customer.
+ **TotalCharges:** Total amount charged to the customer.
+ **Contract:** The type of contract (month-to-month, one year, two years).
+ **Churn:** Whether the customer has churned (Yes/No).

## Tools & Technologies Used
+ **Python:** Programming language used for data analysis and model building.
+ **Pandas & NumPy:** Data manipulation and analysis libraries.
+ **Scikit-learn:** Machine learning library used for model development.
+ **Matplotlib & Seaborn:** Libraries used for data visualization.

## Steps Involved
**Data Preprocessing:**

+ Handle missing values in the TotalCharges column.
+ Convert categorical features into numerical values using label encoding and one-hot encoding.
+ Standardize the feature values for better model performance.

## Model Building:

+ Implement the K-Nearest Neighbors (KNN) algorithm.
+ Train the model using the training dataset.

## Model Evaluation:

+ Evaluate the model using confusion matrix, accuracy.

![image](https://github.com/user-attachments/assets/60e588c2-39d5-4d18-94a9-29759856d5b0)


## Results
Accuracy: The KNN model achieved an accuracy of **93.17%** on the test dataset.

## Conclusion
The KNN classifier performed reasonably well in predicting customer churn. This project demonstrates how customer data can be leveraged to predict churn and potentially improve customer retention strategies.

## Future Work
+ Experiment with other classification algorithms such as Logistic Regression, Decision Trees, and Random Forest.
+ Improve model performance by tuning hyperparameters.
+ Visualize key insights and feature importance using tools like Tableau or Power BI.
