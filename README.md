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

## 🚧 Challenges Faced
1. **Understanding the 3-Way Split:** Grasping why we need three separate datasets (training, CV, test) instead of just two, and the role of each in model development.
2. **Bias-Variance Trade-off:** Interpreting the relationship between model complexity, training error, and CV error to diagnose whether a model is underfitting or overfitting.
3. **Regularization Parameter Tuning:** Understanding how increasing λ (regularization) reduces overfitting but can lead to underfitting if set too high.
4. **Learning Curves Interpretation:** Understanding that increasing training data helps high variance models but does not improve high bias models.
5. **Neural Network Architecture Design:** Choosing appropriate layer sizes and activations for classification problems.
6. **Classification Error Calculation:** Implementing the function to count incorrect predictions and compute the fraction of errors.
7. **Regularization in Keras:** Using `kernel_regularizer=tf.keras.regularizers.l2(0.1)` in Dense layers.

## 📋 Important Submission Notes (AutoGrader)
To avoid grader errors from the DeepLearning.AI AutoGrader, I made sure NOT to:
- Add any extra `print` statements in the assignment
- Add any extra code cells
- Change any of the function parameters
- Use global variables inside graded exercises (unless specifically instructed)
- Change the assignment code where not required

> 💡 **Tips:**
> - Use `# grade-up-to-here` in graded cells to check progress
> - Do not edit or delete non-graded cells
> - Save and submit the notebook for grading

## 🛠️ Technologies & Tools
- Python 3
- NumPy
- Matplotlib
- scikit-learn (LinearRegression, Ridge, train_test_split, PolynomialFeatures)
- TensorFlow/Keras (Sequential, Dense, regularizers)
- Custom utility functions (assigment_utils.py)

## 📈 Results

### Polynomial Regression Results
**Optimal Degree:** Found using CV error minimization
- Training error decreases as degree increases (as expected)
- CV error decreases initially, then increases (overfitting)
- Optimal degree at the minimum of CV error curve

**Regularization Tuning:**
- λ = 0 → High variance (overfitting)
- Moderate λ → Good generalization
- Large λ → High bias (underfitting)

**Learning Curves:**
- More training data improves high variance models
- Does NOT significantly improve high bias models

### Neural Network Results

| Model | Architecture | Training Error | CV Error | Analysis |
|-------|--------------|----------------|----------|----------|
| **Complex** | 3 layers (120, 40, 6) | ~0.005 | ~0.05 | Overfitting (high variance) |
| **Simple** | 2 layers (6, 6) | ~0.08 | ~0.10 | Moderate bias/variance |
| **Regularized** | 3 layers + L2(0.1) | ~0.06 | ~0.08 | Good generalization |

**Best Model:** Regularized complex model with λ = 0.1
- Balanced performance on training and CV sets
- Comparable to "ideal" model performance
- Test performance consistent with CV performance

## 🔗 Key Concepts & Formulas
- **Mean Squared Error (Regression):** `J = (1/(2m)) * Σ(yhat - y)²`
- **Classification Error:** `J = (1/m) * Σ(yhat ≠ y)`
- **L2 Regularization:** Adds `λ * Σ(w²)` to the loss function
- **Bias-Variance Trade-off:** 
  - High Bias (Underfitting): Model is too simple
  - High Variance (Overfitting): Model is too complex
- **Data Splitting:** Training (60%), Cross-Validation (20%), Test (20%)
- **Learning Curve:** Training and CV error as function of training set size

## 🧪 Implementation Highlights
```python
# Mean Squared Error
def eval_mse(y, yhat):
    m = len(y)
    err = sum((yhat[i] - y[i])**2 for i in range(m))
    return err / (2 * m)
