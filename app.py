#import streamlit as st
import pandas as pd
from model import classification_models as md

print("Select classification models")
print("Press \n1 for Logistic Regression \n"+
             "2 for Decision Tree \n" +
             "3 for KNN   \n" +
             "4 for Naive Bayes \n" +
             "5 for Random Forest \n" +
             "6 for XGBoost \n" +
             "7 for All above")
try:            
    input_model = int(input("Enter value from 1 to 7: "))

    if input_model > 7 or input_model < 1 :
        print("wrong input given !!!!!!")
        exit(0)
except :
    print("invalid input ")
    exit(0)
             

md.train_test_as_per_input(input_model)

#X_train, y_train, X_test, y_test = md.load_data_windows()
#X_train, y_train, X_test, y_test = md.encode_data(X_train, y_train, X_test, y_test)
#models = md.initialize_models()
#results = []
#results = md.train_and_test_all_models(models, X_train, y_train, X_test, y_test)
#results = md.pridict_and_evaluate_model(models, X_train, y_train, X_test, y_test)
#md.print_results(results)






