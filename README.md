# Classifiers Performance Comparison on Imbalanced Target Data

#### Jupyter Notebook with code, experimentation, future insights:[here](https://github.com/agneflores/Classification-Model-Comparison-Project/blob/main/Model_Comparison_KNN_LR_DT_SVM.ipynb)

## Objective

To evaluate and compare the performance of K-Nearest Neighbors (KNN), Logistic Regression, Decision Tree, and Support Vector Machine (SVM) models on a dataset with an imbalanced target variable. The analysis focused on how each model's precision, recall, and overall accuracy metrics were affected before and after applying the SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance. The objective of the classification task is to predict whether a client will subscribe to a term deposit (variable y). 

## Data
The data pertains to direct marketing campaigns conducted by a Portuguese banking institution, which were executed via phone calls. In many cases, multiple contacts with the same client were necessary to determine whether they would subscribe to a bank term deposit.

There are two datasets available: 
1) bank-full.csv, containing all instances, organized by date (from May 2008 to November 2010)
2) bank.csv, which includes 10% of the instances (4,521), randomly selected from bank-full.csv. The smaller dataset is intended for testing more computationally intensive machine learning algorithms (e.g., SVM).

The objective of the classification task is to predict whether a client will subscribe to a term deposit (variable y).

Number of Instances: 45,211 in bank-full.csv and 4,521 in bank.csv.
Number of Attributes: 16 + the target.

For the purpose of this project the bank-full data set is used.

Both datasets can be found and downloaded here: https://archive.ics.uci.edu/dataset/222/bank+marketing

#### Data Attributes: 

#### Bank client data:

age (numeric)

job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services") 

marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)

education (categorical: "unknown","secondary","primary","tertiary")

default: has credit in default? (binary: "yes","no")

balance: average yearly balance, in euros (numeric)

housing: has housing loan? (binary: "yes","no")

loan: has personal loan? (binary: "yes","no")

#### Related with the last contact of the current campaign:

contact: contact communication type (categorical: "unknown","telephone","cellular")

day: last contact day of the month (numeric)

month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")

duration: last contact duration, in seconds (numeric)

#### Other attributes:

campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)

pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)

previous: number of contacts performed before this campaign and for this client (numeric)

poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

#### Target variable:

y - has the client subscribed a term deposit? (binary: "yes","no")

Missing Attribute Values: None

<img width="1230" alt="image" src="https://github.com/user-attachments/assets/6cbbcdeb-db54-4a5f-a59a-35a4d43b0956">

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Modeling and Performance

Performance measures overview:

#### Precision:

Definition: Precision measures the proportion of true positive predictions among all positive predictions made by the model. It tells you how many of the predicted positives were actually correct.

Formula: Precision = True Positives / (True Positives + False Positives)

Interpretation: High precision indicates that when the model predicts a positive outcome, it is likely to be correct.

#### Recall:

Definition: Recall, also known as sensitivity or true positive rate, measures the proportion of actual positives that were correctly identified by the model.

Formula: Recall = True Positives / (True Positives + False Negatives)

Interpretation: High recall indicates that the model is good at capturing most of the positive cases.

#### F1-Score:

Definition: The F1-score is the harmonic mean of precision and recall, providing a single metric that balances both concerns. It's useful when you want a balance between precision and recall.

Formula: F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

Interpretation: A high F1-score means the model has a good balance between precision and recall.

#### Support:

Definition: Support is the number of actual occurrences of each class in the dataset. It represents the number of instances for each label.

Interpretation: Support helps understand how much weight each class has in the evaluation of the performance metrics.

#### Accuracy:

Definition: Accuracy measures the proportion of correctly predicted instances (both true positives and true negatives) among the total number of instances.

Formula: Accuracy = (True Positives + True Negatives) / Total Instances

Interpretation: Accuracy gives an overall indication of how often the model is correct, but it can be misleading in the case of imbalanced datasets.

#### Macro Average (Macro Avg):

Definition: Macro average calculates the average of a metric (like precision, recall, or F1-score) across all classes, treating each class equally regardless of its support.

Interpretation: Macro average is useful for understanding how the model performs across all classes, giving equal importance to each class.

#### Weighted Average (Weighted Avg):

Definition: Weighted average calculates the average of a metric across all classes, but it takes into account the support (number of instances) for each class. Larger classes contribute more to the weighted average.

Interpretation: Weighted average provides a more balanced view of the model's performance, especially in imbalanced datasets where some classes have more instances than others.


These performance measures together give a comprehensive view of a model's effectiveness, helping to evaluate not only its overall accuracy but also how well it handles different classes, particularly in cases of imbalanced data.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### K-Nearest Neighbors (n=3)
<img width="800" alt="image" src="https://github.com/user-attachments/assets/19721cef-2862-4e28-8c4e-3d9d14e53f14">

### Summary:

#### Improvement for Class 1:
SMOTE successfully improved the model's ability to detect class 1 (minority class), as evidenced by the significant increase in recall from 0.35 to 0.63. This means the model is now better at identifying true positives for the minority class.

#### Trade-offs:
While recall for class 1 improved, precision for class 1 decreased, leading to more false positives. This trade-off is common when applying SMOTE, as the model becomes more inclusive in identifying the minority class, sometimes at the cost of precision.

#### Overall Balance:
The macro average metrics show that the model's ability to handle both classes more equitably improved, with better recall but slightly lower precision.

### Conclusion:
After applying SMOTE, the KNN model became more effective at identifying the minority class (class 1), as seen by the improved recall and F1-score for class 1. However, this came at the cost of reduced precision for class 1 and a slight drop in overall accuracy. The application of SMOTE made the model more balanced in handling both classes, which is particularly useful in scenarios where detecting the minority class is crucial.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Logistic Regression with balanced class weights adjustments 
<img width="800" alt="image" src="https://github.com/user-attachments/assets/34b4a2ab-7dc9-47fa-9307-61a4d8b2411d">

### Summary:

#### Class Imbalance Handling:
The model shows a decent recall for both classes, particularly for the minority class (class 1). This indicates that the logistic regression model is reasonably sensitive to class 1, which is a good sign for imbalanced datasets. However, the low precision for class 1 (0.42) indicates that many instances predicted as class 1 are actually class 0, leading to a higher number of false positives.

#### Model Performance:
The high recall and low precision for class 1 suggest that the model is more concerned with capturing all possible positive instances, even at the cost of increasing false positives. This trade-off might be acceptable depending on the application; for example, in cases where it's more critical to identify all positives (class 1) even if some negatives (class 0) are incorrectly identified as positives.

The overall accuracy of 0.85 is strong but might not fully reflect the model's performance on the minority class.

### Potential Improvements:

#### Threshold Adjustment: 
Adjusting the decision threshold for class 1 could help improve precision at the cost of recall. This would be useful if the application requires higher confidence in positive predictions.

#### Alternative Models: 
Exploring more complex models such as Random Forests or Gradient Boosting Machines (GBMs) that might handle imbalanced datasets more effectively.

#### Class Weights: 
Logistic regression allows for class weights. You might consider fine-tuning the class weights to improve precision for the minority class.

### Conclusion:
The logistic regression model demonstrates strong overall recall, especially for the minority class. However, the low precision for class 1 indicates a relatively high occurrence of false positives. While this trade-off might be acceptable depending on the application, scenarios where false positives are costly may require further tuning or more advanced techniques to enhance the model's performance.

Applying SMOTE did not alter the model's performance results, as the class_weight='balanced' parameter in Logistic Regression had already effectively addressed the class imbalance. The lack of improvement after applying SMOTE suggests that the model was already well-optimized for the imbalanced dataset using class weights. This underscores that class_weight='balanced' can be a highly effective tool in Logistic Regression for managing imbalanced datasets, often making additional techniques like SMOTE unnecessary.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Decision Tree Classifier
Below is the table comparing the results of the Decision Tree Classifier with balanced class weights adjustments to the results after applying SMOTE on the training data instead of adjusting for class weight balance.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/083f8c11-7405-4a69-b113-d5a056f920e7">

### Summary:

#### Improvement for Class 1: 
The recall for class 1 improved significantly from 0.45 to 0.53 after SMOTE was applied, meaning that the model became better at identifying the minority class. This is a positive outcome of using SMOTE.

#### Trade-offs: 
The improvement in recall for class 1 came with a slight decrease in precision for class 1, indicating more false positives. Additionally, the slight decrease in recall for class 0 is a typical trade-off when balancing an imbalanced dataset.

#### Overall Balance: 
The macro average recall improved, which indicates that the model became more balanced in its ability to detect both classes. However, the overall accuracy and precision for class 1 slightly decreased, which is a typical consequence of applying SMOTE.

### Conclusion:
After applying SMOTE, the Decision Tree model became more effective at identifying the minority class (class 1) by improving its recall. However, this improvement came with a slight decrease in precision for class 1 and a marginal drop in overall accuracy. The model now has a better balance in handling both classes, making it more suitable for situations where identifying the minority class is crucial, even if it comes at the cost of some overall accuracy.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Support Vector Machine
Below is the table comparing the results of the Support Vector Machine model with balanced class weights to the results after applying SMOTE on the training data instead of adjusting for class weight balance.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/58a37b95-94d6-49e6-867b-ae1bcf1bbdea">

### Summary:

#### Improvement for Class 0:
The model's recall for class 0 slightly improved after SMOTE, resulting in a small increase in F1-score. Precision remained high, but with a very slight decrease.

#### Class 1 Stability:
The performance metrics for class 1 (minority class) remained relatively stable, with only minor changes in precision and recall. This suggests that SMOTE did not significantly alter the model's ability to predict class 1, which could be due to how the SVM algorithm handles the synthetic data.

#### Overall Stability:
The overall metrics (accuracy, macro average, and weighted average) remained stable, with minor improvements in some areas. The model continued to perform well after applying SMOTE, but the expected significant improvements in recall for class 1 were not observed.

### Conclusion:
Applying SMOTE to the SVM model resulted in slight improvements in recall for class 0 and a slight increase in overall accuracy. However, the expected improvements in recall for class 1 were not as pronounced, and the precision for class 1 remained similar. The SVM model appears to be robust to the application of SMOTE, maintaining strong overall performance while handling the synthetic data effectively. The overall impact of SMOTE on this SVM model was minimal, indicating that the model was already performing well on the imbalanced dataset.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Executive Summary & Conclusion
Based on the analysis of the results before and after applying SMOTE for the KNN, Decision Tree, and SVM models (and considering that Logistic Regression results did not change significantly), here’s a summary to help decide which model is the best:

#### K-Nearest Neighbors (KNN)
Before SMOTE: Class 1 Recall: 0.35 (Low) Class 1 Precision: 0.55 (Moderate) Overall Accuracy: 0.89
After SMOTE: Class 1 Recall: 0.63 (Significant Improvement) Class 1 Precision: 0.41 (Decreased) Overall Accuracy: 0.85 (Slight Decrease)

#### Logistic Regression
Before SMOTE: Class 1 Recall: 0.82 (High) Class 1 Precision: 0.42 (Moderate) Overall Accuracy: 0.85
After SMOTE: Class 1 Recall: 0.82 (Identical) Class 1 Precision: 0.42 (Identical) Overall Accuracy: 0.85 (Identical)

#### Decision Tree
Before SMOTE: Class 1 Recall: 0.45 (Moderate) Class 1 Precision: 0.47 (Moderate) Overall Accuracy: 0.87
After SMOTE: Class 1 Recall: 0.53 (Moderate Improvement) Class 1 Precision: 0.42 (Slight Decrease) Overall Accuracy: 0.86 (Slight Decrease)

#### Support Vector Machine (SVM)
Before SMOTE: Class 1 Recall: 0.86 (High) Class 1 Precision: 0.43 (Moderate) Overall Accuracy: 0.85
After SMOTE: Class 1 Recall: 0.79 (Slight Decrease) Class 1 Precision: 0.44 (Slight Increase) Overall Accuracy: 0.86 (Slight Increase)

### Conclusion Summary:

When comparing the performance of K-Nearest Neighbors (KNN), Logistic Regression, Decision Tree, and Support Vector Machine (SVM) models on imbalanced data, we observed that the application of SMOTE had varying impacts across the models.

K-Nearest Neighbors (KNN) showed a significant improvement in recall for class 1 after applying SMOTE, but this came at the cost of decreased precision and overall accuracy. While the model became more sensitive to the minority class, it also became more prone to false positives.

Logistic Regression maintained consistent performance before and after applying SMOTE, with high recall for class 1 and moderate precision. The identical results suggest that the class_weight='balanced' parameter effectively handled the class imbalance, making SMOTE redundant for this model.

Decision Tree experienced a moderate improvement in recall after SMOTE was applied, but this was offset by a slight decrease in precision and overall accuracy. The model's sensitivity to the minority class improved, but it became less precise in identifying true positives.

Support Vector Machine (SVM) had the highest recall for class 1 before SMOTE, with a slight decrease in recall and a marginal increase in precision after SMOTE. The overall accuracy improved slightly, indicating that SVM maintained robust performance even with the application of SMOTE.

### Best Model Selection:
Support Vector Machine (SVM) is the best model for this scenario.

### Justification:

#### High Initial Recall: 
SVM already exhibited the highest recall for class 1 before applying SMOTE, indicating strong performance in identifying the minority class.
#### Balanced Performance After SMOTE: 
While there was a slight decrease in recall after SMOTE, SVM still maintained a high level of recall with a slight increase in precision and overall accuracy. This balance makes it a reliable choice for scenarios where both sensitivity to the minority class and overall accuracy are critical.
#### Robustness: 
SVM's ability to maintain strong performance with minimal changes after SMOTE demonstrates its robustness in handling imbalanced datasets effectively without significant degradation in other metrics.

Given these considerations, the SVM model offers the best combination of high recall, balanced precision, and stable overall accuracy, making it the most suitable choice for dealing with imbalanced target data.
