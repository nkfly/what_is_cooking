import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, RandomizedPCA
import xgboost as xgb


#from keras.models import Sequential
#from keras.layers.core import Dense, Dropout
#from nltk.stem.wordnet import WordNetLemmatizer
import re
import csv
import itertools
import os.path
import json
import math
from datetime import datetime


def read_data_by_column(filename):
	tripType, visitNumber, weekDay, scanCount, departmentDescription, finelineNumber = [], [], [], [], [], []
	with open(filename, 'r') as f:
		spamreader = csv.DictReader(f, delimiter=',', quotechar='"')
		header = spamreader.fieldnames
		if len(header) == 7:
			for line in spamreader:
				tripType.append(line['TripType'])
				visitNumber.append(line['VisitNumber'])
				weekDay.append(line['Weekday'])
				scanCount.append(int(line['ScanCount']))
				departmentDescription.append(line['DepartmentDescription'])
				finelineNumber.append(line['FinelineNumber'])
			return tripType, visitNumber, weekDay, scanCount, departmentDescription, finelineNumber

		else:
			for line in spamreader:
				visitNumber.append(line['VisitNumber'])
				weekDay.append(line['Weekday'])
				scanCount.append(int(line['ScanCount']))
				departmentDescription.append(line['DepartmentDescription'])
				finelineNumber.append(line['FinelineNumber'])
	return visitNumber, weekDay, scanCount, departmentDescription, finelineNumber


def category_to_k_hot():
	train_tripType, train_visitNumber, train_weekDay, train_scanCount, train_departmentDescription, train_FinelineNumber = read_data_by_column("train.csv")
	test_visitNumber, test_weekDay, test_scanCount, test_departmentDescription, test_FinelineNumber = read_data_by_column("test.csv")

	test_descount2visitnum = {}
	test_finecount2visitnum = {}
	for i in xrange(len(test_visitNumber)):
		if test_visitNumber[i] not in test_descount2visitnum:
			test_descount2visitnum[test_visitNumber[i]] = abs(test_scanCount[i])
		else:
			test_descount2visitnum[test_visitNumber[i]] += abs(test_scanCount[i])
		if test_visitNumber[i] not in test_finecount2visitnum:
			test_finecount2visitnum[test_visitNumber[i]] = abs(test_scanCount[i])
		else:
			test_finecount2visitnum[test_visitNumber[i]] += abs(test_scanCount[i])
	j = json.dumps(test_descount2visitnum, indent=4)
	f = open('walmart_data/test_descount2visitnum.json', 'w')
	print >> f, j
	f.close()
	j = json.dumps(test_finecount2visitnum, indent=4)
	f = open('walmart_data/test_finecount2visitnum.json', 'w')
	print >> f, j
	f.close()

	train_descount2visitnum = {}
	train_finecount2visitnum = {}
	for i in xrange(len(train_visitNumber)):
		if train_visitNumber[i] not in train_descount2visitnum:
			train_descount2visitnum[train_visitNumber[i]] = abs(train_scanCount[i])
		else:
			train_descount2visitnum[train_visitNumber[i]] += abs(train_scanCount[i])
		if train_visitNumber[i] not in train_finecount2visitnum:
			train_finecount2visitnum[train_visitNumber[i]] = abs(train_scanCount[i])
		else:
			train_finecount2visitnum[train_visitNumber[i]] += abs(train_scanCount[i])
	j = json.dumps(train_descount2visitnum, indent=4)
	f = open('walmart_data/train_descount2visitnum.json', 'w')
	print >> f, j
	f.close()
	j = json.dumps(train_finecount2visitnum, indent=4)
	f = open('walmart_data/train_finecount2visitnum.json', 'w')
	print >> f, j
	f.close()

	"""
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
	"""


def loaddict(filename):
	with open('walmart_data/' + filename + '.json') as data_file:
		datadict = json.load(data_file)
	return datadict


def idf():
	trainData = loaddict('train')
	testData = loaddict('test')

	dL = loaddict('departmentDescription_r')
	wL = loaddict('weekDay_r')

	dL = len(dL)
	wL = len(wL)

	train_idf = {}
	for k1, v1 in trainData.iteritems():
		for k2, v2 in v1.iteritems():
			if k2 not in train_idf:
				train_idf[k2] = 1
			else: train_idf[k2] += 1
	for k in train_idf:
		train_idf[k] = math.log10(1+float(len(trainData))/train_idf[k])

	test_idf = {}
	for k1, v1 in testData.iteritems():
		for k2, v2 in v1.iteritems():
			if k2 not in test_idf:
				test_idf[k2] = 1
			else: test_idf[k2] += 1
	for k in test_idf:
		test_idf[k] = math.log10(1+float(len(testData))/test_idf[k])

	j = json.dumps(train_idf, indent=4)
	f = open('walmart_data/train_idf.json', 'w')
	print >> f, j
	f.close()

	j = json.dumps(test_idf, indent=4)
	f = open('walmart_data/test_idf.json', 'w')
	print >> f, j
	f.close()


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
		spamreader = csv.DictReader(f, delimiter=',', quotechar='"')
		header = spamreader.fieldnames
		if len(header) == 7:
			for line in spamreader:
				tripType = line['TripType']
				visitNumber = line['VisitNumber']
				weekDay = line['Weekday']
				scanCount = int(line['ScanCount'])
				departmentDescription = line['DepartmentDescription']
				finelineNumber = line['FinelineNumber']
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
			for line in spamreader:
				visitNumber = line['VisitNumber']
				weekDay = line['Weekday']
				scanCount = int(line['ScanCount'])
				departmentDescription = line['DepartmentDescription']
				finelineNumber = line['FinelineNumber']
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


def buyNumEncoder(num):
	if num <= 20:
		return num
	elif num <= 50:
		return 20+1
	elif num <= 100:
		return 20+2
	elif num <= 200:
		return 20+3
	else: return 20+4


def train_json2matrix():
	visitNumber2tripType = loaddict('visitNumber2tripType')
	train_finecount2visitnum = loaddict('train_finecount2visitnum')
	train_descount2visitnum = loaddict('train_descount2visitnum')
	train_idf = loaddict('train_idf')

	with open('walmart_data/train.json') as data_file:
		trainData = json.load(data_file)
	
	answer = []
	data = []
	row = []
	col = []
	count = 0
	for k1, v1 in trainData.items():
		r = count
		returnOrNot = False
		buyNum = 0
		depCount = 0
		lineCount = 0
		for k2, v2 in v1.items():
			""" 1/0, weighting(0, 1, 1)
			if int(k2) < 7:
				data.append(float(1)*0)
			elif int(k2) in range(7, 69+7):
				data.append(float(1)*1)
			else:
				data.append(float(1)*1)
			"""
			""" tfidf
			if int(k2) in range(7, 69+7):
				data.append(float(abs(v2))/float(train_descount2visitnum[k1] + train_finecount2visitnum[k1] + 1) * train_idf[k2])
			elif int(k2) >= 76:
				data.append(float(abs(v2))/float(train_descount2visitnum[k1] + train_finecount2visitnum[k1] + 1) * train_idf[k2])
			else:
				data.append(1/float(train_descount2visitnum[k1] + train_finecount2visitnum[k1] + 1) * train_idf[k2])
			"""
			"""tfidf + finelineNumber
			if int(k2) >= 76:
				data.append(float(v2)/float(train_finecount2visitnum[k1]) * train_idf[k2])
				row.append(r)
				col.append(int(k2)-76)
			"""
			data.append(1)
			row.append(r)
			col.append(int(k2))
			buyNum += abs(int(v2))
			if int(v2) < 0:
				returnOrNot = True
			if int(k2) in range(7, 69+7):
				depCount += 1
			elif int(k2) >= 76:
				lineCount += 1
		if returnOrNot:
			data.append(2)
			row.append(r)
			col.append(7+69+5354)
		else:
			data.append(2)
			row.append(r)
			col.append(7+69+5354+1)
		data.append(2)
		row.append(r)
		col.append(7+69+5354+2+buyNumEncoder(buyNum)-1)
		data.append(1)
		row.append(r)
		col.append(7+69+5354+2+24+depCount-1)
		data.append(1)
		row.append(r)
		col.append(7+69+5354+2+24+69+lineCount-1)

		answer.append(visitNumber2tripType[k1])
		count += 1
	# Create the COO-matrix
	coo = coo_matrix((data,(row,col)), shape=(len(trainData), 7+69+5354+2+24+69+5354))
	# Let Scipy convert COO to CSR format and return
	return csr_matrix(coo), answer

def test_json2matrix():
	test_finecount2visitnum = loaddict('test_finecount2visitnum')
	test_descount2visitnum = loaddict('test_descount2visitnum')
	test_idf = loaddict('test_idf')

	with open('walmart_data/test.json') as data_file:
		testData = json.load(data_file)
	
	id = []
	data = []
	row = []
	col = []
	count = 0
	for k1, v1 in testData.items():
		r = count
		returnOrNot = False
		buyNum = 0
		depCount = 0
		lineCount = 0
		for k2, v2 in v1.items():
			""" 1/0, weighting(0, 1, 1)
			if int(k2) < 7:
				data.append(float(1)*0)
			elif int(k2) in range(7, 69+7):
				data.append(float(1)*1)
			else:
				data.append(float(1)*1)
			"""
			""" tf #idf
			if int(k2) in range(7, 69+7):
				data.append(float(abs(v2))/float(test_descount2visitnum[k1]+test_finecount2visitnum[k1]+1) * test_idf[k2])
			elif int(k2) >= 76:
				data.append(float(abs(v2))/float(test_descount2visitnum[k1]+test_finecount2visitnum[k1]+1) * test_idf[k2])
			else:
				data.append(1/float(test_descount2visitnum[k1]+test_finecount2visitnum[k1]+1) * test_idf[k2])
			"""
			"""tfidf + finelineNumber
			if int(k2) >= 76:
				data.append(float(abs(v2))/float(test_finecount2visitnum[k1]) * test_idf[k2])
				row.append(r)
				col.append(int(k2)-76)
			"""
			data.append(1)
			row.append(r)
			col.append(int(k2))
			buyNum += abs(int(v2))
			if int(k2) in range(7, 69+7):
				depCount += 1
			if int(v2) < 0:
				returnOrNot = True
			elif int(k2) >= 76:
				lineCount += 1
		if returnOrNot:
			data.append(2)
			row.append(r)
			col.append(7+69+5354)
		else:
			data.append(2)
			row.append(r)
			col.append(7+69+5354+1)
		data.append(2)
		row.append(r)
		col.append(7+69+5354+2+buyNumEncoder(buyNum)-1)
		data.append(1)
		row.append(r)
		col.append(7+69+5354+2+24+depCount-1)
		data.append(1)
		row.append(r)
		col.append(7+69+5354+2+24+69+lineCount-1)

		id.append(k1)
		count += 1
	# Create the COO-matrix
	coo = coo_matrix((data,(row,col)), shape=(len(testData), 7+69+5354+2+24+69+5354))
	# Let Scipy convert COO to CSR format and return
	return csr_matrix(coo), id


"""
Unique Number
 
 - train_departmentDescription - 69
 - test_departmentDescription - 69 (all appear in train_departmentDescription)

 - train_tripType - 38

 - train_FinelineNumber - ?
 - test_FinelineNumber - ?
 - total FinelineNumber - 5354

"""
if __name__ == '__main__':
	#category_to_k_hot()
	#csv2json()
	#idf()

	print "--prepare training data--"
	train_X, train_y = train_json2matrix()

	print "--prepare testing data--"
	test_X, test_id = test_json2matrix()
	test_id = [int(d) for d in test_id]

	print train_X.shape, len(train_y)
	print test_X.shape, len(test_id)

	tripType2class = loaddict('tripType_r')
	
	train_X = train_X
	train_Y = [tripType2class[train_y[i]] for i in range(len(train_y))]

	test_X = test_X
	test_Y = [0 for i in range(test_X.shape[0])]


	xg_train = xgb.DMatrix(train_X, label=train_Y)
	xg_test = xgb.DMatrix(test_X, label=test_Y)


	param = {}
	# use softmax multi-class classification
	param['objective'] = 'multi:softprob'
	# scale weight of positive examples
	param['eta'] = 0.1
	param['max_depth'] = 10
	param['silent'] = 0
	param['nthread'] = 8
	param['num_class'] = 38
	param['eval_metric'] = 'mlogloss'


	watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
	num_round = 500
	bst = xgb.train(param, xg_train, num_round, watchlist )

	result = bst.predict( xg_test )

	with open('walmart_data/tripType.json', 'r') as f:
		class2tripType = json.load(f)
	headerOrder = []

	for i in range(38):
		headerOrder.append('"TripType_' + class2tripType[str(i)] + '"')

	
	with open('walmart_data/planswer2.csv', 'w') as w:
		w.write('"VisitNumber",')
		w.write(",".join(headerOrder) + '\n')
		result = [b for a,b in sorted(zip(test_id, result))]
		test_id = sorted(test_id)
		for i in xrange(len(result)):
			w.write(str(test_id[i])+',')
			w.write(','.join([ str(r) for r in result[i]]))
			w.write('\n')	

	"""
	pca = PCA(n_components=500)
	print '--- pca starting ---'
	train_X = pca.fit_transform(train_X.todense(), train_y)
	test_X = pca.transform(test_X.todense())
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
	"""
