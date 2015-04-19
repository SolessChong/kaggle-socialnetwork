import sklearn
import sklearn.svm
import sklearn.grid_search
import sklearn.metrics
import numpy as np
import csv

def load_data(train_fn, test_fn):
	content = np.loadtxt(train_fn, delimiter=",", skiprows=1)
	label = content[:,0:1]
	data = content[:,1:]
	content = np.loadtxt(test_fn, delimiter=",", skiprows=1)
	test = content

	return data, label, test

def preprocess(data, label, test):
	def transform(x):
		return np.log(1+x)

	data = np.c_[transform(data[:,0:11])-transform(data[:,11:22])]
	test = np.c_[transform(test[:,0:11])-transform(test[:,11:22])]

	perm = np.random.permutation(len(data))
	data = data[perm, :]
	label = label[perm, :]
 
	return data, label, test

def write_data(fn, pred):
	with open(fn, 'wb') as csvfile:
		writer = csv.writer(csvfile, delimiter=",")
		writer.writerow(["Id", "Choice"])
		for i in range(len(pred)):
			writer.writerow([i+1, pred[i]])

def predict(model, test):
	rst = model.predict_proba(test)[:,1]

	return rst

def solve(clf, data, label):
	clf.fit(data, label)

	return clf

if __name__ == "__main__":
	data, label, test = load_data('data/train.csv', 'data/test.csv')
	data, label, test = preprocess(data, label, test)

	model = sklearn.linear_model.LogisticRegression(fit_intercept=False)
	model = solve(model, data, label)

	p_train = model.predict_proba(data)
	p_train = p_train[:,1:2]
	print 'AuC score on training data:', sklearn.metrics.roc_auc_score(label, p_train)

	rst = predict(model, test)

	write_data('submission/LogisticRegression_diff_log.csv', rst)