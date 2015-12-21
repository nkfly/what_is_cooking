import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.linear_model import LogisticRegression

#from keras.models import Sequential
#from keras.layers.core import Dense, Dropout
#from nltk.stem.wordnet import WordNetLemmatizer
import re
import itertools
import os.path
import json
from datetime import datetime


def read_data_by_column(filename):
	tripType, visitNumber, weekDay, scanCount, departmentDescription, FinelineNumber = [], [], [], [], [], []
	with open(filename, 'r') as f:
		header = f.readline().strip().split(',')
		if len(header) == 7:
			for line in f:
				entries = line.strip().split(',')
				tripType.append(entries[0])
				visitNumber.append(entries[1])
				weekDay.append(entries[2])
				scanCount.append(int(entries[4]))
				departmentDescription.append(entries[5])
				FinelineNumber.append(entries[6])
			return tripType, visitNumber, weekDay, scanCount, departmentDescription, FinelineNumber

		else:
			for line in f:
				entries = line.strip().split(',')
				visitNumber.append(entries[0])
				weekDay.append(entries[1])
				scanCount.append(int(entries[3]))
				departmentDescription.append(entries[4])
				FinelineNumber.append(entries[5])
	return visitNumber, weekDay, scanCount, departmentDescription, FinelineNumber


def category_to_k_hot():
	train_tripType, train_visitNumber, train_weekDay, train_scanCount, train_departmentDescription, train_FinelineNumber = read_data_by_column("train.csv")
	test_visitNumber, test_weekDay, test_scanCount, test_departmentDescription, test_FinelineNumber = read_data_by_column("test.csv")

	visitNumber2tripType = {}
	print len(train_tripType), len(train_visitNumber)
	if len(train_tripType) == len(train_visitNumber):
		for i in xrange(len(train_tripType)):
			visitNumber2tripType[train_visitNumber[i]] = train_tripType[i]
	j = json.dumps(visitNumber2tripType, indent=4)
	f = open('walmart_data/visitNumber2tripType.json', 'w')
	print >> f, j
	f.close()

	le = LabelEncoder()
	le.fit_transform(train_departmentDescription)
	train_departmentDescription = list(le.classes_)
	departmentDescription = {}
	departmentDescription_r = {}
	for i in xrange(len(train_departmentDescription)):
		departmentDescription[i] = train_departmentDescription[i]
		departmentDescription_r[train_departmentDescription[i]] = i
	j = json.dumps(departmentDescription, indent=4)
	f = open('walmart_data/departmentDescription.json', 'w')
	print >> f, j
	f.close()
	j = json.dumps(departmentDescription_r, indent=4)
	f = open('walmart_data/departmentDescription_r.json', 'w')
	print >> f, j
	f.close()

	le.fit_transform(train_weekDay)
	train_weekDay = list(le.classes_)
	weekDay = {}
	weekDay_r = {}
	for i in xrange(len(train_weekDay)):
		weekDay[i] = train_weekDay[i]
		weekDay_r[train_weekDay[i]] = i
	j = json.dumps(weekDay, indent=4)
	f = open('walmart_data/weekDay.json', 'w')
	print >> f, j
	f.close()
	j = json.dumps(weekDay_r, indent=4)
	f = open('walmart_data/weekDay_r.json', 'w')
	print >> f, j
	f.close()

	train_FinelineNumber.extend(test_FinelineNumber)
	le.fit_transform(train_FinelineNumber)
	total_FinelineNumber = list(le.classes_)
	finelineNumber = {}
	finelineNumber_r = {}
	for i in xrange(len(total_FinelineNumber)):
		finelineNumber[i] = total_FinelineNumber[i]
		finelineNumber_r[total_FinelineNumber[i]] = i
	j = json.dumps(finelineNumber, indent=4)
	f = open('walmart_data/finelineNumber.json', 'w')
	print >> f, j
	f.close()
	j = json.dumps(finelineNumber_r, indent=4)
	f = open('walmart_data/finelineNumber_r.json', 'w')
	print >> f, j
	f.close()

	le.fit_transform(train_tripType)
	train_tripType = list(le.classes_)
	tripType = {}
	tripType_r = {}
	for i in xrange(len(train_tripType)):
		tripType[i] = train_tripType[i]
		tripType_r[train_tripType[i]] = i
	j = json.dumps(tripType, indent=4)
	f = open('walmart_data/tripType.json', 'w')
	print >> f, j
	f.close()
	j = json.dumps(tripType_r, indent=4)
	f = open('walmart_data/tripType_r.json', 'w')
	print >> f, j
	f.close()


def loaddict(filename):
	with open('walmart_data/' + filename + '.json') as data_file:
		datadict = json.load(data_file)
	return datadict


def read_data_to_dict(filename):
	departmentDescription_r = loaddict('departmentDescription_r')
	finelineNumber_r = loaddict('finelineNumber_r')
	tripType_r = loaddict('tripType_r')
	weekDay_r = loaddict('weekDay_r')
	visitNumber2tripType = loaddict('visitNumber2tripType')
	dL = len(departmentDescription_r)
	wL = len(weekDay_r)

	datadict = {}
	with open(filename, 'r') as f:
		header = f.readline().strip().split(',')
		if len(header) == 7:
			for line in f:
				entries = line.strip().split(',')
				tripType = entries[0]
				visitNumber = entries[1]
				weekDay = entries[2]
				scanCount = int(entries[4])
				departmentDescription = entries[5]
				finelineNumber = entries[6]
				if visitNumber in datadict:
					if departmentDescription_r[departmentDescription] + wL in datadict[visitNumber]:
						datadict[visitNumber][departmentDescription_r[departmentDescription] + wL] += scanCount
					else:
						datadict[visitNumber][departmentDescription_r[departmentDescription] + wL] = scanCount
					if finelineNumber_r[finelineNumber] + wL + dL in datadict[visitNumber]:
						datadict[visitNumber][finelineNumber_r[finelineNumber] + wL + dL] += scanCount
					else:
						datadict[visitNumber][finelineNumber_r[finelineNumber] + wL + dL] = scanCount
				else:
					datadict[visitNumber] = {weekDay_r[weekDay]: 1, departmentDescription_r[departmentDescription] + wL: scanCount, finelineNumber_r[finelineNumber] + wL + dL: scanCount}
		else:
			for line in f:
				entries = line.strip().split(',')
				visitNumber = entries[0]
				weekDay = entries[1]
				scanCount = int(entries[3])
				departmentDescription = entries[4]
				finelineNumber = entries[5]
				if visitNumber in datadict:
					if departmentDescription_r[departmentDescription] + wL in datadict[visitNumber]:
						datadict[visitNumber][departmentDescription_r[departmentDescription] + wL] += scanCount
					else:
						datadict[visitNumber][departmentDescription_r[departmentDescription] + wL] = scanCount
					if finelineNumber_r[finelineNumber] + wL + dL in datadict[visitNumber]:
						datadict[visitNumber][finelineNumber_r[finelineNumber] + wL + dL] += scanCount
					else:
						datadict[visitNumber][finelineNumber_r[finelineNumber] + wL + dL] = scanCount
				else:
					datadict[visitNumber] = {weekDay_r[weekDay]: 1, departmentDescription_r[departmentDescription] + wL: scanCount, finelineNumber_r[finelineNumber] + wL + dL: scanCount}
		return datadict


def csv2json():
	train = read_data_to_dict('train.csv')
	j = json.dumps(train, indent=4)
	f = open('walmart_data/train.json', 'w')
	print >> f, j
	f.close()

	test = read_data_to_dict('test.csv')
	j = json.dumps(test, indent=4)
	f = open('walmart_data/test.json', 'w')
	print >> f, j
	f.close()

def train_json2matrix():
	visitNumber2tripType = loaddict('visitNumber2tripType')

	with open('walmart_data/train.json') as data_file:
		trainData = json.load(data_file)
	
	answer = []
	data = []
	row = []
	col = []
	count = 0
	for k1, v1 in trainData.items():
		r = count
		for k2, v2 in v1.items():
			data.append(1)
			row.append(r)
			col.append(k2)
		answer.append(visitNumber2tripType[k1])
		count += 1
	# Create the COO-matrix
	coo = coo_matrix((data,(row,col)), shape=(len(trainData), 7+69+5290))
	# Let Scipy convert COO to CSR format and return
	return csr_matrix(coo), answer

def test_json2matrix():
	with open('walmart_data/test.json') as data_file:
		testData = json.load(data_file)
	
	id = []
	data = []
	row = []
	col = []
	count = 0
	for k1, v1 in testData.items():
		r = count
		for k2, v2 in v1.items():
			data.append(1)
			row.append(r)
			col.append(k2)
		id.append(k1)
		count += 1
	# Create the COO-matrix
	coo = coo_matrix((data,(row,col)), shape=(len(testData), 7+69+5290))
	# Let Scipy convert COO to CSR format and return
	return csr_matrix(coo), id


"""
Unique Number
 
 - train_departmentDescription - 69
 - test_departmentDescription - 68 (all appear in train_departmentDescription)

 - train_tripType - 38

 - train_FinelineNumber - 5132
 - test_FinelineNumber - 5139
 - total FinelineNumber - 5290

"""
if __name__ == '__main__':
	#category_to_k_hot()
	#csv2json()

	train_X, train_y = train_json2matrix()
	test_X, test_id = test_json2matrix()
	test_id = [int(d) for d in test_id]
	print train_X.shape, len(train_y)
	print test_X.shape, len(test_id)

	clf = LogisticRegression()
	clf.fit(train_X, train_y)
	result = clf.predict_proba(test_X)
	classes = list(clf.classes_)
	classes = [int(c) for c in classes]
	print classes
	print result[0]
	result = [ [b for a,b in sorted(zip(classes, r))] for r in result]
	print result[0]
	
	with open('walmart_data/answer.csv', 'w') as w:
		w.write('"VisitNumber","TripType_3","TripType_4","TripType_5","TripType_6","TripType_7","TripType_8","TripType_9","TripType_12","TripType_14","TripType_15","TripType_18","TripType_19","TripType_20","TripType_21","TripType_22","TripType_23","TripType_24","TripType_25","TripType_26","TripType_27","TripType_28","TripType_29","TripType_30","TripType_31","TripType_32","TripType_33","TripType_34","TripType_35","TripType_36","TripType_37","TripType_38","TripType_39","TripType_40","TripType_41","TripType_42","TripType_43","TripType_44","TripType_999"\n')
		result = [b for a,b in sorted(zip(test_id, result))]
		test_id = sorted(test_id)
		for i in xrange(len(result)):
			w.write(str(test_id[i]) + ',')
			w.write(','.join([ str(r) for r in result[i]]))
			w.write('\n')	
