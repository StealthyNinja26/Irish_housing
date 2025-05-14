# 🏠 Ireland House Price Analysis & Prediction

This project is a comprehensive data science portfolio piece focused on analyzing and predicting house prices across Ireland using real-world data. Leveraging a 2024 housing dataset from Kaggle, it applies exploratory data analysis (EDA), classification, regression, and a user-friendly Streamlit application to showcase the insights and predictions. The aim is to provide both technical and practical value in understanding property trends and forecasting home values using machine learning.

---

## 📁 Project Structure

```
├── eda_irish_housing.ipynb        # EDA, data cleaning, feature engineering
├── classification_model.pkl       # Trained classification model (joblib)
├── regression_model.pkl           # Trained regression model (joblib)
├── app.py                         # Streamlit GUI application
├── ireland-house-properties-dataset-2024.csv # Raw dataset from Kaggle
└── README.md                      # This file
```

Each component plays a distinct role in building the pipeline from raw data to interactive prediction tools. The notebooks explore the data and prepare it for model training, while the `.pkl` files store the trained models. The Streamlit app (`app.py`) ties everything together into a smooth, graphical experience.

---

## 📊 Dataset Overview

* **Source:** [Kaggle - Ireland House Properties Dataset 2024](https://www.kaggle.com/datasets/adnankhalid007/ireland-house-properties-dataset-2024)
* **Size:** Thousands of listings for properties sold or listed across various counties in Ireland.
* **Columns include:**

  * Property Type (Apartment, Detached, etc.)
  * Bedrooms, Bathrooms
  * County (Dublin, Galway, Cork, etc.)
  * BER Rating (Energy efficiency)
  * Size (in square meters)
  * Parking availability, Sale Type (new or second-hand)
  * Price (target variable)

This rich dataset allows both regression and classification tasks, offering a versatile platform for modeling.

---

## 🧠 Machine Learning Models

### 1. 🏷️ Classification

* **Goal:** Predict a price category label such as Low, Medium-Low, Medium, Medium-High, or High.
* **Method:** Custom ordinal encoding to reflect economic ordering of price categories. Trained using a supervised learning algorithm on preprocessed features including size, location, amenities, and BER.

### 2. 💶 Regression

* **Goal:** Predict the actual price of a home in euros.
* **Method:** Standard regression pipeline using numeric and encoded categorical features. It captures trends in housing size, region, and property type to estimate market value.

---

## 🚀 Streamlit Web App

The Streamlit app serves as an accessible way for users to interact with the trained models. It has two primary functionalities:

* **📊 Predict Price Category:**

  * Users fill in a property’s attributes and get a prediction on which price tier the house would likely fall into.

* **💶 Predict Price Value:**

  * Using the same inputs, the regression model estimates a euro value for the house based on learned trends in the training data.

### ▶️ To Run the App Locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

This launches a local server where users can interactively test different housing scenarios.

### 🌐 Example UI

<img width="743" alt="Irish_housing_gui" src="https://github.com/user-attachments/assets/aab310f2-1b9c-4021-89fb-041f25a99378" />

---

## 🛠️ Features & Techniques

* Data Cleaning: Addressing missing values, data types, and erroneous outliers
* Feature Engineering: One-hot encoding, ordinal categorization, boolean transformations
* Model Training: Scikit-learn pipelines with train/test splits and validation
* Model Persistence: Export and reuse trained models using `joblib`
* Streamlit Interface: A responsive, clean UI built with Streamlit widgets to gather user input
* Custom Category Decoding: Manual mapping used instead of `LabelEncoder` for more controlled results

These steps ensure a robust pipeline that bridges raw data to usable prediction tools.

---

## ✅ Requirements

```
pandas
numpy
scikit-learn
streamlit
joblib
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## 📌 Future Enhancements

* 🔍 Add SHAP or other model interpretability techniques for feature importance
* ☁️ Deploy to Streamlit Cloud for public access
* 📈 Include economic indicators or time trends to enhance forecasting
* 💾 Add a database backend for storing and querying predictions

---

## 🙋‍♂️ Author

Created by **Praveen S. Sundaram** as part of a professional data science portfolio to demonstrate practical, full-cycle machine learning application development — from raw data and EDA to modeling and interactive delivery.
