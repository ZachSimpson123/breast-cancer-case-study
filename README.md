# Breast Cancer Prediction — Case Study (DLBDSME01)

Interpretable machine learning model to classify breast tumors as malignant or benign,
built for DLBDSME01 – Model Engineering (Task 1).

## Files

- `breast_cancer_case_study.py` — the full analysis: data exploration, model training, and interpretation
- `Date_Zach_Simpson_32212726_DLBDSME01.docx` — the written case study
- `figures/` — generated automatically when you run the script (charts and plots)
- `models/` — generated automatically when you run the script (trained model + results)

## How to Run

1. Install the required packages:
   ```
   pip install numpy pandas matplotlib seaborn scikit-learn shap joblib
   ```

2. Run the script:
   ```
   python breast_cancer_case_study.py
   ```

3. Check the new `figures/` and `models/` folders for the output.

## What It Does

1. Loads the Breast Cancer Wisconsin dataset and checks its quality
2. Splits and scales the data
3. Trains benchmark models, then three main candidates: Logistic Regression, Decision Tree, and Random Forest
4. Tunes each model with cross-validation
5. Explains the models using coefficients, feature importance, SHAP, and partial dependence plots
6. Evaluates errors and saves the final model

## Result

Logistic Regression achieved the best balance of accuracy (F1 = 0.964) and interpretability,
and is the model recommended for deployment.