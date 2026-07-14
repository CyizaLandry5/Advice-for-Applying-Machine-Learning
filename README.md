# Assignment: Advice for Applying Machine Learning

## Course Information
- **Course:** Machine Learning (Advanced Learning Algorithms)
- **Student:** Mpayimana Cyixa Landry
- **School:** DeepLearning.AI
- **Assignment:** Advice for Applying Machine Learning

## 📌 Overview
This lab explored essential techniques for **evaluating and improving machine learning models**. I learned how to diagnose whether a model is suffering from **high bias (underfitting)** or **high variance (overfitting)**, and how to apply strategies like regularization, increasing training data, and adjusting model complexity to improve performance.

The lab covered both **polynomial regression** and **neural network** models, providing hands-on experience with splitting data into training, cross-validation, and test sets, calculating error metrics, and tuning hyperparameters.

## 🎯 Learned
- **Data Splitting:** How to properly divide data into Training, Cross-Validation (CV), and Test sets (60/20/20 split).
- **Error Metrics:** 
  - Mean Squared Error (MSE) for regression problems
  - Classification error for categorical problems
- **Bias vs. Variance Diagnosis:** Using training and CV error to identify underfitting vs. overfitting:
  - High training error + high CV error → High bias (underfitting)
  - Low training error + high CV error → High variance (overfitting)
- **Model Complexity Tuning:** Finding the optimal polynomial degree by comparing training and CV errors.
- **Regularization Tuning:** Using L2 regularization to reduce overfitting and finding the optimal λ (lambda) value.
- **Learning Curves:** Understanding how increasing the training set size (m) affects model performance.
- **Neural Network Evaluation:** Building and evaluating neural network models with different architectures (complex vs. simple).
- **Regularization in Neural Networks:** Applying L2 regularization to neural networks to prevent overfitting.

## ⚙️ Key Skills Practiced
- Splitting data with `train_test_split` from scikit-learn
- Implementing MSE and classification error functions
- Polynomial regression with scikit-learn
- Neural network construction with TensorFlow/Keras
- Regularization with L2 (weight decay)
- Model evaluation on training, CV, and test sets
- Hyperparameter tuning (degree, λ, architecture)
- Visualization of model performance curves
