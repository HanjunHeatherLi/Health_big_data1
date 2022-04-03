import utils
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
#Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

'''
You may generate your own features over here.
Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients. Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''
def my_features():
	#TODO: complete this
	#training dataset
	X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	#testing data featurize
	filepath = '../data/test/'
	events = pd.read_csv(filepath + 'events.csv')
	feature_map = pd.read_csv(filepath + 'event_feature_map.csv')
	#aggregate_events
	test_events=events[["patient_id","event_id","value"]]
	idxed_event = pd.merge(test_events, feature_map, how='left', on='event_id').drop(["event_id"],axis=1).dropna()

	feature_value = idxed_event.groupby(["patient_id", "idx"]).count().reset_index()
	feature_value_max = feature_value.drop(["patient_id"], axis=1).groupby("idx").agg({'value':[np.max]}).reset_index()
	#print(feature_value_max)
	aggregated_events = pd.merge(feature_value, feature_value_max, how='left', on='idx')
	aggregated_events["feature_value"] = (aggregated_events["value"]) / (aggregated_events[("value", "amax")])
	aggregated_events = aggregated_events.rename(columns={"idx": "feature_id"})
	aggregated_events = aggregated_events[['patient_id', 'feature_id', 'feature_value']]
	aggregated_events["total"] = aggregated_events.apply(lambda row: [int(row["feature_id"]), row["feature_value"]],axis=1)
	aggregated_events.drop(["feature_id", "feature_value"], axis=1, inplace=True)
	aggregated_events = aggregated_events.groupby('patient_id')['total'].apply(list).to_frame()
	patient_features = {}
	for index, row in aggregated_events.iterrows():
		patient_features[index] = row["total"]
	##save_svmlight
	deliverable2 = open("../deliverables/test_features.txt", 'wb')
	sorted_patient_features = sorted(patient_features.items())
	#print(sorted_patient_features[0])
	for key, item in sorted_patient_features:
		item = sorted(item)
		deliverable2.write(bytes(str(int(key)) + " " + " ".join(
			str(":".join([str(int(i[0])), str(format(i[1], '.6f'))])) for i in item) + " " + "\r\n", 'UTF-8'))
	deliverable2.close()
	X_test,_ = utils.get_data_from_svmlight("../deliverables/test_features.txt")
	return X_train,Y_train,X_test

'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''

def my_classifier_predictions(X_train,Y_train,X_test):
	#TODO: complete this
	from sklearn.metrics import roc_auc_score
	x_simu, y_simu = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")

	#Support vector machine
	"""svm_para = {'kernel': ('linear', 'rbf'), 'C': [0.001,0.01, 0.1, 1, 10]}
	svm_search=GridSearchCV(SVC(),param_grid=svm_para,scoring='roc_auc', cv=5, n_jobs=-1).fit(X_train, Y_train)
	y_pred_svm=svm_search.predict(x_simu)
	print("---SVM---")
	print(svm_search.best_params_)
	#print(roc_auc_score(y_simu, svm_search.decision_function(x_simu)))
	print(svm_search.best_score_)
	print(svm_search.score(x_simu, y_simu))"""

	# logistical regression
	""""
	LR_para = {'penalty': ['l1','l2', 'elasticnet'], 'C': [0.01,0.1,1,10,100], "solver": ["newton-cg","lbfgs", "liblinear", "sag", "saga"]}
	LR_search = GridSearchCV(LogisticRegression(max_iter = 1000), param_grid=LR_para, scoring='roc_auc', cv=5, n_jobs=-1).fit(X_train, Y_train)
	y_pred_LR = LR_search.predict(x_simu)
	print("---Logistic Regression---")
	print(LR_search.best_params_)
	print(LR_search.best_score_)
	print(LR_search.score(x_simu, y_simu))

	# Key nearest neighbor
	KNN_para={'n_neighbors':[3,5,11,19], 'weights':['uniform','distance'], 'metric':['euclidean','manhattan']}
	KNN_search=GridSearchCV(KNeighborsClassifier(),param_grid=KNN_para, scoring='roc_auc', cv=5, n_jobs=-1).fit(X_train, Y_train)
	y_pred_KNN = KNN_search.predict(x_simu)
	print("---Key nearest neighbor---")
	print(KNN_search.best_params_)
	print(KNN_search.best_score_)"""
	"""
	#Random Forest
	RF_para={'n_estimators': [200, 500, 1000, 1500, 2000],'criterion': ['gini', 'entropy'],'max_depth':list(range(5,20))}
	RF_search = GridSearchCV(RandomForestClassifier(), param_grid=RF_para, scoring='roc_auc', cv=5, n_jobs=-1).fit(X_train, Y_train)
	y_pred_RF = RF_search.predict(x_simu)
	print("---Random Forest---")
	print(RF_search.best_params_)
	print(RF_search.best_score_)
	print(RF_search.score(x_simu, y_simu))"""
	# Decision tree
	DT_para = {'criterion': ['gini', 'entropy'],'splitter':['best','random'],'max_depth':list(range(5,20)),'random_state':list(range(0,20)) }
	DT_search = GridSearchCV(DecisionTreeClassifier(), param_grid=DT_para, scoring='roc_auc', cv=6, n_jobs=-1).fit(
		X_train, Y_train)
	y_pred_RF = DT_search.predict(x_simu)
	print("---Decision Tree---")
	print(DT_search.best_params_)
	print(DT_search.best_score_)
	print(DT_search.score(x_simu, y_simu))

	return y_pred_RF

def myadaboost(X_train,Y_train,X_test):

	#clf = LogisticRegression(max_iter = 1000, C=1, penalty = "l2", solver="liblinear" )
	#clf = RandomForestClassifier(n_estimators=1000, criterion='entropy',max_depth=17)
	clf = DecisionTreeClassifier(criterion = 'entropy',max_depth=12, random_state=4)
	ada_clf = AdaBoostClassifier(base_estimator = clf)
	ada_para = {"n_estimators":[2000,5000,8000], "learning_rate" : [0.01, 0.1,0.5,1], 'algorithm': ["SAMME","SAMME.R"],'random_state':[0,4,7,19]}
	grid_ada = GridSearchCV(ada_clf, param_grid=ada_para, scoring='roc_auc', cv=6, n_jobs=-1).fit(
		X_train, Y_train)
	y_pred_ada = grid_ada.predict(X_test)
	print("---adaboost---")
	print(grid_ada.best_params_)
	print(grid_ada.best_score_)
	#print(grid_ada.score(X_train, X_train))
	return y_pred_ada

def mygradientboost(X_train,Y_train,X_test):
	gb_para = {"n_estimators": [1000, 3000, 5000, 8000], "learning_rate": [0.01, 0.1, 1], "max_depth" :[8,12,16,20,24,28,32]}
	grid_ada = GridSearchCV(GradientBoostingClassifier(), param_grid=gb_para, scoring='roc_auc', cv=6, n_jobs=-1).fit(
		X_train, Y_train)
	y_pred_ada = grid_ada.predict(X_test)
	print("---gradient boost---")
	print(grid_ada.best_params_)
	print(grid_ada.best_score_)
	# print(grid_ada.score(X_train, X_train))
	return y_pred_ada


def main():
	X_train, Y_train, X_test = my_features()
	Y_pred = myadaboost(X_train,Y_train,X_test)
	#Y_pred =my_classifier_predictions(X_train,Y_train,X_test)
	#Y_pred = mygradientboost(X_train, Y_train, X_test)
	utils.generate_submission("../deliverables/test_features.txt",Y_pred)
	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.

if __name__ == "__main__":
    main()

