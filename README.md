# ⚙️ Hyperparameter Optimization using Optuna, GridSearchCV & RandomizedSearchCV

## 📌 Project Overview

This project demonstrates advanced hyperparameter tuning techniques to improve machine learning model performance.
An XGBoost classifier is trained on a classification dataset and optimized using three different tuning methods:

* Optuna (Bayesian Optimization)
* GridSearchCV
* RandomizedSearchCV

The goal is to compare optimization strategies and evaluate how hyperparameter tuning enhances model accuracy and ROC-AUC performance.

---

## 🚀 Key Features

* Data preprocessing and feature scaling
* Baseline XGBoost model training
* Hyperparameter tuning using **Optuna**
* Traditional tuning using **GridSearchCV**
* Efficient tuning using **RandomizedSearchCV**
* Performance comparison of all approaches
* Evaluation using Accuracy and ROC-AUC

---

## 🛠️ Technologies Used

* Python
* Scikit-learn
* XGBoost
* Optuna
* Pandas & NumPy

---

## 📊 Workflow

1. Load and preprocess dataset
2. Train baseline XGBoost model
3. Evaluate baseline performance
4. Optimize hyperparameters using Optuna
5. Train and evaluate Optuna-tuned model
6. Perform GridSearchCV tuning
7. Perform RandomizedSearchCV tuning
8. Compare all results

---

## 📈 Evaluation Metrics

* Accuracy Score
* ROC-AUC Score
* Cross-validation performance

These metrics help determine the effectiveness of different hyperparameter optimization techniques.

---

## 📂 Project Structure

```
Hyperparameter-Optimization-Optuna-vs-GridSearch
│── hyperparameter_optimization.py
│── requirements.txt
│── README.md
```

---

## ▶️ How to Run the Project

### 1️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Run the script

```
python hyperparameter_optimization.py
```

---

## 🧠 Learning Outcomes

* Understanding importance of hyperparameter tuning
* Using Optuna for advanced optimization
* Comparing Grid Search vs Random Search vs Bayesian optimization
* Improving model performance using tuning techniques
* Building a structured ML optimization workflow

---

## 🔮 Future Improvements

* Add visualization of Optuna trials
* Use cross-validation inside Optuna objective
* Apply tuning on real-world datasets
* Deploy optimized model as API or web app

---

## 👨‍💻 Author

**Prateek Manjunath**</br>
Machine Learning & Data Science Enthusiast
