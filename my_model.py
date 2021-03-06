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


'''
input:
output: X_train,Y_train,X_test
'''
def my_features():
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

input: X_train, Y_train, X_test
output: Y_pred
'''

def my_classifier_predictions(X_train,Y_train,X_test):
	from sklearn.metrics import roc_auc_score
	x_simu, y_simu = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")

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

	clf = DecisionTreeClassifier(criterion = 'entropy',max_depth=12, random_state=4)
	ada_clf = AdaBoostClassifier(base_estimator = clf)
	ada_para = {"n_estimators":[2000,5000,8000], "learning_rate" : [0.01, 0.1,0.5,1], 'algorithm': ["SAMME","SAMME.R"],'random_state':[0,4,7,19]}
	grid_ada = GridSearchCV(ada_clf, param_grid=ada_para, scoring='roc_auc', cv=6, n_jobs=-1).fit(
		X_train, Y_train)
	y_pred_ada = grid_ada.predict(X_test)
	print("---adaboost---")
	print(grid_ada.best_params_)
	print(grid_ada.best_score_)
	return y_pred_ada

def mygradientboost(X_train,Y_train,X_test):
	gb_para = {"n_estimators": [1000, 3000, 5000, 8000], "learning_rate": [0.01, 0.1, 1], "max_depth" :[8,12,16,20,24,28,32]}
	grid_ada = GridSearchCV(GradientBoostingClassifier(), param_grid=gb_para, scoring='roc_auc', cv=6, n_jobs=-1).fit(
		X_train, Y_train)
	y_pred_ada = grid_ada.predict(X_test)
	print("---gradient boost---")
	print(grid_ada.best_params_)
	print(grid_ada.best_score_)
	return y_pred_ada


def main():
	X_train, Y_train, X_test = my_features()
	Y_pred = myadaboost(X_train,Y_train,X_test)
	utils.generate_submission("../deliverables/test_features.txt",Y_pred)
	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.

if __name__ == "__main__":
    main()

