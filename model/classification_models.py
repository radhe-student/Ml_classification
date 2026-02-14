
# Step 1: Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

model_options = ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost", "All"]

def load_data_windows():
    #load data set
    #X_train = pd.read_csv("Dataset\\train\\X_train.txt", delim_whitespace=True, header=None)
    X_train = pd.read_csv("Dataset\\train\\X_train.txt", sep="\\s+",  header=None)
    y_train = pd.read_csv("Dataset\\train\\y_train.txt", header=None)
    #X_test = pd.read_csv("Dataset\\test\\X_test.txt", delim_whitespace=True, header=None)
    X_test = pd.read_csv("Dataset\\test\\X_test.txt", sep="\\s+", header=None)
    y_test = pd.read_csv("Dataset\\test\\y_test.txt", header=None)
    #print(X_train.shape)
    #print(X_test.shape)
    return X_train, y_train, X_test, y_test
    

def load_data():
    #load data set
    #X_train = pd.read_csv("UCI HAR Dataset\\train\\X_train.txt", delim_whitespace=True, header=None)
    X_train = pd.read_csv("Dataset/train/X_train.txt", sep="\\s+",  header=None)
    y_train = pd.read_csv("Dataset/train/y_train.txt", header=None)
    #X_test = pd.read_csv("Dataset\\test\\X_test.txt", delim_whitespace=True, header=None)
    X_test = pd.read_csv("Dataset/test/X_test.txt", sep="\\s+", header=None)
    y_test = pd.read_csv("Dataset/test/y_test.txt", header=None)
    #print(X_train.shape)
    #print(X_test.shape)
    return X_train, y_train, X_test, y_test


 


def encode_data(X_train, y_train, X_test, y_test):
    #encode the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train.values.ravel())
    y_test = le.transform(y_test.values.ravel())
    return X_train, y_train, X_test, y_test


def initialize_models():
    #prepare the model list
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": xgb.XGBClassifier(eval_metric='mlogloss')
    }
    return models




def pridict_and_evaluate_model(models, name, X_train, y_train, X_test, y_test, results):

    if name in models.keys():
        #results = []
        models[name].fit(X_train, y_train)
        y_pred = models[name].predict(X_test)

        # For multiclass AUC, use 'ovr'
        try:
            y_prob = models[name].predict_proba(X_test)
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        except:
            auc = None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_test, y_pred)

        results.append([name, acc, auc, prec, rec, f1, mcc])

        return results
    else:
        print("No implementation for model: ", name)
        return []
		
		
		



def train_and_test_all_models(models, X_train, y_train, X_test, y_test):
    # Train th model, predict the data, and evaluate
    results = []
    
    for name, model in models.items():
        
        results = pridict_and_evaluate_model(models, name, X_train, y_train, X_test, y_test, results)
        
    return results
    
def train_test_as_per_input(input_val):
    print("Selected Model: ", model_options[input_val-1])
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = encode_data(X_train, y_train, X_test, y_test)
    models = initialize_models()
    results = []
    if(input_val == len(model_options)):
        results = train_and_test_all_models(models, X_train, y_train, X_test, y_test)
    else:
        results = pridict_and_evaluate_model(models, model_options[input_val-1], X_train, y_train, X_test, y_test, results)
        
    print_results(results)
    
    
    

def print_results(results):
    #Display results
    
    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"])
    
    print(results_df)
    
