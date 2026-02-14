
**Problem Statement:**  Implement multiple classification models

**Dataset description:** UCI Human Activity Recognition database: Human Activity Recognition database built from the recordings of 30 subjects performing 
activities of daily living while carrying a waist-mounted smartphone with embedded inertial sensors.

it has 10299 data set with 561 features.
No of test dataset: 7352
No of training dataset: 2947


**Models used:**
	"Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
	
**Comparison Table with the evaluation metrics calculated for models.**
	
| Model					| Accuracy	| AUC		| Precision | Recall	| F1 Score	| MCC      |
| Logistic Regression	| 0.954191	| 0.997532	| 0.956045	| 0.954191	| 0.954144	| 0.945333 |
| Decision Tree			| 0.849678	| 0.908037	| 0.850325	| 0.849678	| 0.849189	| 0.819627 |
| KNN					| 0.880217	| 0.975445	| 0.888331	| 0.880217	| 0.879025	| 0.857806 |
| Naive Bayes			| 0.770275	| 0.957861	| 0.794683	| 0.770275	| 0.76877	| 0.728609 |
| Random Forest			| 0.923312	| 0.994713	| 0.924298	| 0.923312	| 0.922965	| 0.908099 |
| XGBoost				| 0.938242	| 0.996948	| 0.939523	| 0.938242	| 0.93802	| 0.926098 |


**ML Model Name Observation about model performance**
Logistic Regression  | AUC is high so it is classifing the data with high accurately It is the best performing model
Decision Tree        | it is the fifth best performing model
kNN                  | it is the fourth best performing model
Naive Bayes          | it is the sixth best performing model
Random Forest        | it is the third best performing model, it is classifing the data with high accurately
XGBoost              | AUC is high so it is classifing the data with high accurately. it is the second best performing model



 
 
