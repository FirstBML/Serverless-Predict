import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore', category=ConvergenceWarning)


# Configuration
C = 1.0
N_SPLITS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 1
OUTPUT_FILE = f'model_C={C}.bin'

# Feature definitions
NUMERICAL = ['tenure', 'monthlycharges', 'totalcharges']
CATEGORICAL = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]

url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"

def load_and_prepare_data(url):
    """Load and prepare the dataset."""
    print("Loading data...")
    df = pd.read_csv(url)
    
    # Clean column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Identify and clean categorical columns
    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')
    
    # Fix numerical columns with errors
    df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
    df.totalcharges = df.totalcharges.fillna(0)
    
    # Convert target to binary
    df.churn = (df.churn == 'yes').astype(int)
    
    print(f"Data loaded: {len(df)} rows")
    return df


def train(df_train, y_train, C=1.0):
    """Train the logistic regression model."""
    dicts = df_train[CATEGORICAL + NUMERICAL].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model


def predict(df, dv, model):
    """Make predictions using the trained model."""
    dicts = df[CATEGORICAL + NUMERICAL].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred


def cross_validate(df_full_train, C, n_splits):
    """Perform cross-validation."""
    print(f"\nPerforming {n_splits}-fold cross-validation...")
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(df_full_train), 1):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]
        
        y_train = df_train.churn.values
        y_val = df_val.churn.values
        
        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)
        
        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)
        print(f"Fold {fold}: AUC = {auc:.3f}")
    
    mean_auc = np.mean(scores)
    std_auc = np.std(scores)
    print(f'\nC={C} | Mean AUC: {mean_auc:.3f} +- {std_auc:.3f}')
    return mean_auc, std_auc


def save_model(dv, model, output_file):
    """Save the trained model and vectorizer."""
    with open(output_file, 'wb') as f_out:
        pickle.dump((dv, model), f_out)
    print(f"Model saved to {output_file}")


def main():
    # Load and prepare data
    url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = load_and_prepare_data(url)
    
    # Split into train and test
    df_full_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Training set: {len(df_full_train)} rows")
    print(f"Test set: {len(df_test)} rows")
    
    # Perform cross-validation
    cross_validate(df_full_train, C, N_SPLITS)
    
    # Train final model on full training set
    print("\nTraining final model...")
    dv, model = train(df_full_train, df_full_train.churn.values, C=C)
    
    # Evaluate on test set
    y_pred = predict(df_test, dv, model)
    y_test = df_test.churn.values
    test_auc = roc_auc_score(y_test, y_pred)
    print(f"Test AUC: {test_auc:.3f}")
    
    # Save the model
    save_model(dv, model, OUTPUT_FILE)


if __name__ == "__main__":
    main()