# Mobile Price Classification Dashboard


## a. Problem Statement
 The objective of this assignment is to develop an end-to-end machine learning workflow.  
 This involves selecting a classification dataset, implementing six distinct machine learning models (Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, and XGBoost), 
 evaluating them using multiple performance metrics, and deploying the solution as an interactive Streamlit application.

## b. Dataset Description
* **Dataset Name**: Mobile Price Classification.
*  **Source**: Public repository (Kaggle/UCI).
*  **Instance Size**: 2,000 samples (Minimum required: 500).
*  **Feature Size**: 20 features including RAM, Battery Power, and Pixel Dimensions (Minimum required: 12).
* **Target Variable**: `price_range` (4 classes: 0-Low Cost, 1-Medium Cost, 2-High Cost, 3-Very High Cost).

## c. Models Used
All six models were trained and evaluated on the same dataset using a 28% test split.

### Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 95.00% | 0.998 | 0.952 | 0.950 | 0.950 | 0.934 |
| Decision Tree | 84.11% | 0.930 | 0.844 | 0.841 | 0.842 | 0.788 |
| kNN | 57.32% | 0.797 | 0.584 | 0.573 | 0.571 | 0.437 |
| Naive Bayes | 81.07% | 0.954 | 0.815 | 0.811 | 0.812 | 0.747 |
| Random Forest (Ensemble) | 87.86% | 0.980 | 0.880 | 0.879 | 0.878 | 0.839 |
| XGBoost (Ensemble) | 90.36% | 0.985 | 0.905 | 0.904 | 0.904 | 0.872 |

### Observations on Model Performance 
| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | **Best Performer**: Achieved the highest accuracy (95.00%), indicating a very strong linear relationship between RAM and price classes. |
| Decision Tree | **Stable Baseline**: Captured non-linear patterns well but was prone to lower recall compared to ensemble methods. |
| kNN | **Weakest Performer**: The low accuracy (57.32%) suggests that simple distance metrics are less effective in this 20-dimensional feature space. |
| Naive Bayes | **Strong Separator**: Despite simple assumptions, it maintained a high AUC (0.954), showing it can distinguish between price classes effectively. |
| Random Forest (Ensemble) | **Robust Ensemble**: Successfully reduced variance of individual trees, providing very balanced precision and recall scores. |
| XGBoost (Ensemble) | **Highly Efficient**: Second-best model (90.36%); the gradient boosting mechanism effectively minimized errors through iterative learning. |
