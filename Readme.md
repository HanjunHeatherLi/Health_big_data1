# Mortality Prediction
### Overview

Using data from MIMIC database, medical treatment during 2000 days before index date as features to predict if patients will be die on day 30 or not.

For index date: die patients- 30 days before death; alive patients-last medical event day
## Computing basic statistics
Event count: Number of events recorded for a given patient. Note that every line in
the input file is an event.

Encounter count: Count of unique dates on which a given patient visited the hospital.
All the events - DIAG, LAB and DRUG - should be considered as hospital visiting
events.

Record length: Duration (in number of days) between the first event and last event
for a given patient.
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

SVMLight: target feature:value feature:value ... feature:value

Example: <br/>
1 2:0.5 3:0.12 10:0.9 2000:0.3 <br/>
0 4:1.0 78:0.6 1009:0.2
## Constructing models 
Logistic regression, Decision Tree, SVM, Random Forest, Adaboost

Logistic regression gives high performance on testing data (accuracy 0.74 and AUC 0.74)
