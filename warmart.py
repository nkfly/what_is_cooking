import json
import math
import operator
import xgboost as xgb
import numpy as np


visitNumber2document = {}
departmentDescription2dimension = {}
tripTypeOrder = ["TripType_3","TripType_4","TripType_5","TripType_6","TripType_7","TripType_8","TripType_9","TripType_12","TripType_14","TripType_15","TripType_18","TripType_19","TripType_20","TripType_21","TripType_22","TripType_23","TripType_24","TripType_25","TripType_26","TripType_27","TripType_28","TripType_29","TripType_30","TripType_31","TripType_32","TripType_33","TripType_34","TripType_35","TripType_36","TripType_37","TripType_38","TripType_39","TripType_40","TripType_41","TripType_42","TripType_43","TripType_44","TripType_999"]
tripType2class = {"TripType_3":0,"TripType_4":1,"TripType_5":2,"TripType_6":3,"TripType_7":4,"TripType_8":5,"TripType_9":6,"TripType_12":7,"TripType_14":8,"TripType_15":9,"TripType_18":10,"TripType_19":11,"TripType_20":12,"TripType_21":13,"TripType_22":14,"TripType_23":15,"TripType_24":16,"TripType_25":17,"TripType_26":18,"TripType_27":19,"TripType_28":20,"TripType_29":21,"TripType_30":22,"TripType_31":23,"TripType_32":24,"TripType_33":25,"TripType_34":26,"TripType_35":27,"TripType_36":28,"TripType_37":29,"TripType_38":30,"TripType_39":31,"TripType_40":32,"TripType_41":33,"TripType_42":34,"TripType_43":35,"TripType_44":36,"TripType_999":37}

with open('train.csv', 'r') as f:
	header = f.readline().strip().split(',')
	for line in f:
		entries = line.strip().split(',')

		tripType = entries[0]
		visitNumber = entries[1]
		weekDay = entries[2]
		scanCount = int(entries[4])
		# departmentDescription = entries[5]
		departmentDescription = entries[6]

		if departmentDescription not in departmentDescription2dimension:
			departmentDescription2dimension[departmentDescription] = len(departmentDescription2dimension)


		if visitNumber not in visitNumber2document:
			visitNumber2document[visitNumber] = {'tripType' : 'TripType_' + tripType, 'weekDay' : weekDay,  departmentDescription : scanCount}
		else:
			if departmentDescription in visitNumber2document[visitNumber]:
				#visitNumber2document[visitNumber][departmentDescription] += scanCount
				visitNumber2document[visitNumber][departmentDescription] = 1
			else:
				visitNumber2document[visitNumber][departmentDescription] = 1
				#visitNumber2document[visitNumber][departmentDescription] = scanCount

X = []
Y = []
day2index = {'"monday"' : 0, '"tuesday"' : 1, '"wednesday"' : 2, '"thursday"' : 3, '"friday"':4, '"saturday"':5, '"sunday"':6}
for visitNumber in visitNumber2document:
	document = visitNumber2document[visitNumber]

	x = [0 for i in range(len(departmentDescription2dimension))]
	week_x = [0 for i in range(7)]
	for key in document:
		if key == 'tripType':
			Y.append(tripType2class[document[key]])
		elif key == 'weekDay':
			week_x[day2index[document[key].lower()]] = 1
			continue
		else:
			x[departmentDescription2dimension[key]] += document[key]
	x.extend(week_x)
	X.append(x)


testVisitNumber2document = {}

with open('test.csv') as f:
	header = f.readline().strip().split(',')
	for line in f:
		entries = line.strip().split(',')

		visitNumber = entries[0]
		weekDay = entries[1]
		scanCount = int(entries[3])
		# departmentDescription = entries[4]
		departmentDescription = entries[5]

		if visitNumber not in testVisitNumber2document:
			testVisitNumber2document[visitNumber] = {'tripType' : 0, 'weekDay' : weekDay,  departmentDescription : scanCount}
		else:
			if departmentDescription in testVisitNumber2document[visitNumber]:
				#testVisitNumber2document[visitNumber][departmentDescription] += scanCount
				testVisitNumber2document[visitNumber][departmentDescription] = 1
			else:
				testVisitNumber2document[visitNumber][departmentDescription] = 1
				#testVisitNumber2document[visitNumber][departmentDescription] = scanCount



test_data = []
test_Y = []
for visitNumber in testVisitNumber2document:
	document = testVisitNumber2document[visitNumber]

	x = [0 for i in range(len(departmentDescription2dimension))]
	week_x = [0 for i in range(7)]
	for key in document:
		if key == 'tripType':
			test_Y.append(0)
		elif key == 'weekDay':
			week_x[day2index[document[key].lower()]] = 1
			continue
		else:
			if key in departmentDescription2dimension:
				x[departmentDescription2dimension[key]] += document[key]
	x.extend(week_x)		
	test_data.append(x)



train_X = X
train_Y = Y

test_X = test_data
test_Y = test_Y


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
num_round = 1500
bst = xgb.train(param, xg_train, num_round, watchlist )

prediction = bst.predict( xg_test )
print prediction
with open('sample_submission.csv') as f:
	header = f.readline()
	with open('warmart_answer.csv', 'w') as w:
		w.write('"VisitNumber","TripType_3","TripType_4","TripType_5","TripType_6","TripType_7","TripType_8","TripType_9","TripType_12","TripType_14","TripType_15","TripType_18","TripType_19","TripType_20","TripType_21","TripType_22","TripType_23","TripType_24","TripType_25","TripType_26","TripType_27","TripType_28","TripType_29","TripType_30","TripType_31","TripType_32","TripType_33","TripType_34","TripType_35","TripType_36","TripType_37","TripType_38","TripType_39","TripType_40","TripType_41","TripType_42","TripType_43","TripType_44","TripType_999"\n')
		for i in range(len(prediction)):
			line = f.readline()
			entries = line.strip().split(',')
			visitNumber = entries[0]

			w.write(visitNumber + ',')
			w.write(','.join([ str(tt) for tt in prediction[i]]))

			w.write('\n')









