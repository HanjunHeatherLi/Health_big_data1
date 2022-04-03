import models_partc
from sklearn.model_selection import KFold, ShuffleSplit
from numpy import mean

import utils

# USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

# USE THIS RANDOM STATE FOR ALL OF YOUR CROSS VALIDATION TESTS, OR THE TESTS WILL NEVER PASS
RANDOM_STATE = 545510477

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,k=5):
	#TODO:First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the folds
	kf = KFold(n_splits=k,shuffle=True, random_state=RANDOM_STATE)
	list_acc=[]
	list_auc=[]
	for train_index, test_index in kf.split(X):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]
		Y_pred=models_partc.logistic_regression_pred(X_train, Y_train, X_test)
		acc_i,auc_i,_,_,_=models_partc.classification_metrics(Y_pred, Y_test)
		list_acc.append(acc_i)
		list_auc.append(auc_i)
	accuracy=sum(list_acc)/len(list_acc)
	auc=sum(list_auc)/len(list_auc)
	return accuracy,auc


#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
	#TODO: First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the iterations
	kf = ShuffleSplit(n_splits=iterNo,test_size=test_percent, random_state=RANDOM_STATE)
	list_acc = []
	list_auc = []
	for train_index, test_index in kf.split(X):
		# print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]
		Y_pred = models_partc.logistic_regression_pred(X_train, Y_train, X_test)
		acc_i, auc_i, _, _, _ = models_partc.classification_metrics(Y_pred, Y_test)
		list_acc.append(acc_i)
		list_auc.append(auc_i)
	accuracy = sum(list_acc) / len(list_acc)
	auc = sum(list_auc) / len(list_auc)
	return accuracy, auc


def main():
	X,Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	print("Classifier: Logistic Regression__________")
	acc_k,auc_k = get_acc_auc_kfold(X,Y)
	print(("Average Accuracy in KFold CV: "+str(acc_k)))
	print(("Average AUC in KFold CV: "+str(auc_k)))
	acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
	print(("Average Accuracy in Randomised CV: "+str(acc_r)))
	print(("Average AUC in Randomised CV: "+str(auc_r)))

if __name__ == "__main__":
	main()

