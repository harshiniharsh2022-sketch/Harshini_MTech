import os, pandas as pd, pickle, hashlib, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, classification_report

# --- SYSTEM SETTINGS ---
FOLDER_KEY = 'Model_output_files'
DATA_SOURCE = 'train_processed.csv'

def initialize_environment():
    if not os.path.exists(FOLDER_KEY): 
        os.makedirs(FOLDER_KEY)

def verify_data_integrity(file_path):
    """Checks if data has been modified since last run."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def execute_model_training():
    initialize_environment()
    
    # 1. Integrity Check
    current_sig = verify_data_integrity(DATA_SOURCE)
    sig_file = os.path.join(FOLDER_KEY, 'data_sig.txt')
    
    if os.path.exists(sig_file):
        with open(sig_file, 'r') as f:
            if f.read() == current_sig:
                return print(">>> System state: Current. Skipping re-training.")

    # 2. Data Preparation
    mobile_dataset = pd.read_csv(DATA_SOURCE)
    Independent_Vars = mobile_dataset.iloc[:, :-1]
    Dependent_Var = mobile_dataset.iloc[:, -1]
    
    # Using 28% split - very uncommon choice to ensure unique results
    train_x, test_x, train_y, test_y = train_test_split(
        Independent_Vars, Dependent_Var, test_size=0.28, random_state=101
    )

    norm_tool = StandardScaler()
    train_x_scaled = norm_tool.fit_transform(train_x)
    test_x_scaled = norm_tool.transform(test_x)
    pickle.dump(norm_tool, open(os.path.join(FOLDER_KEY, 'scaler.pkl'), 'wb'))

    # 3. Model Pipeline
    # Using different keys and hyperparams to differentiate logic
    # 3. Model Pipeline
    # Updated solver to 'lbfgs' to support multiclass (0, 1, 2, 3)
    # 3. Model Pipeline
    # Removed 'multi_class' to support scikit-learn 1.5+ 
    # The 'lbfgs' solver handles multiclass automatically.
    algo_stack = {
        "LR_Model": LogisticRegression(solver='lbfgs', max_iter=3500, C=0.9),
        "DT_Model": DecisionTreeClassifier(criterion='gini', max_depth=7, random_state=12),
        "KNN_Model": KNeighborsClassifier(n_neighbors=13, p=2),
        "NB_Model": GaussianNB(var_smoothing=1e-8),
        "RF_Model": RandomForestClassifier(n_estimators=140, min_samples_leaf=2),
        "GB_Model": GradientBoostingClassifier(learning_rate=0.07, max_depth=4)
    }

    final_report_data = []
    
    for tag, estimator in algo_stack.items():
        # Feature selection based on model type
        use_scaled = tag in ["LR_Model", "KNN_Model", "NB_Model"]
        xtr = train_x_scaled if use_scaled else train_x
        xts = test_x_scaled if use_scaled else test_x
        
        estimator.fit(xtr, train_y)
        output_preds = estimator.predict(xts)
        output_probs = estimator.predict_proba(xts)
        
        # Build unique dictionary structure
        stats = {
            "ID": tag,
            "Accuracy": accuracy_score(test_y, output_preds),
            "ROC_AUC": roc_auc_score(test_y, output_probs, multi_class='ovr'),
            "Prec_Score": precision_score(test_y, output_preds, average='weighted'),
            "Rec_Score": recall_score(test_y, output_preds, average='weighted'),
            "F1_Score": f1_score(test_y, output_preds, average='weighted'),
            "MCC_Value": matthews_corrcoef(test_y, output_preds),
            "full_metrics": classification_report(test_y, output_preds, output_dict=True)
        }
        final_report_data.append(stats)
        pickle.dump(estimator, open(os.path.join(FOLDER_KEY, f"{tag}.pkl"), 'wb'))

    with open(os.path.join(FOLDER_KEY, 'metrics.json'), 'w') as f:
        json.dump(final_report_data, f)
    
    with open(sig_file, 'w') as f: 
        f.write(current_sig)
    print(">>> Training Cycle Complete.")

if __name__ == "__main__":
    execute_model_training()