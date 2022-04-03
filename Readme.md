# Mortality Prediction
### Overview

Using MIMIC database
## Computing basic statistics
## Preparing the data 
### Feature construction
Observation Window: The time interval you will use to identify relevant events. Only
events present in this window should be included while constructing feature vectors.
The size of observation window is 2000 days

Prediction Window: A fixed time interval that is to be used to make the prediction.
Events in this interval should not be included in constructing feature vectors. The size
of prediction window is 30 days

Index date: The day on which mortality is to be predicted. Index date is evaluated as  follows:
– For deceased patients: Index date is 30 days prior to the death date (timestamp field) in data/train/mortality events.csv.
– For alive patients: Index date is the last event date in data/train/events.csv for each alive patient.
### Aggregate events
Vectorization: Aquire feature value pairs(event feature ID, value) for features with the same patient ID, then aggregate features, save in SVMLight Format

## Constructing models 
Logistic regression, Decision Tree, SVM, Random Forest, Adaboost

Logistic regression gives high performance on testing data (accuracy 0.74 and AUC 0.74)
