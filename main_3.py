import json
import math
import operator
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from metric_learn import LMNN
import xgboost as xgb
import numpy as np

def cosineSimilarity(vec1, vec2):
	vec1Len = 0.0

	innerProduct = 0.0
	for dimension in vec1:
		if dimension in vec2:
			innerProduct += vec1[dimension] * vec2[dimension]


		vec1Len += vec1[dimension] * vec1[dimension]


	vec1Len = math.sqrt(vec1Len)


	vec2Len = 0.0
	for dimension in vec2:
		vec2Len += vec2[dimension] * vec2[dimension]

	vec2Len = math.sqrt(vec2Len)

	return innerProduct/(vec1Len * vec2Len)



	

# load train data
with open('train.json') as data_file:
	trainData = json.load(data_file)



ingredient2documentCount = {} # this is prepared to calculate inverted document frequency

ingredient2id = {} # gives ingredient an id

cuisine2id = {}
id2cuisine = {}

for i in xrange(len(trainData)):
	food = trainData[i]

	for j in xrange(len(food['ingredients'])):
		ingredient = food['ingredients'][j].lower().replace('-', '_')
		if ingredient not in ingredient2id:
			ingredient2id[ingredient] = len(ingredient2id) # just give id

		if ingredient not in ingredient2documentCount:
			ingredient2documentCount[ingredient] = 1
		else:
			ingredient2documentCount[ingredient] += 1

		if food['cuisine'] not in cuisine2id:
			cuisine2id[food['cuisine']] = len(cuisine2id)

			id2cuisine[cuisine2id[food['cuisine']]] = food['cuisine']




# now idf is able to be counted
ingredient2idf = {}
for ingredient in ingredient2documentCount:
	ingredient2idf[ingredient] = math.log10(1+len(trainData)/ingredient2documentCount[ingredient])




id2vector = []
# id2cuisine = {} # the trainId to cuisine(class)
Y = []
for i in xrange(len(trainData)):
	food = trainData[i]

	vector = {}
	for j in xrange(len(food['ingredients'])):
		ingredient = food['ingredients'][j].lower().replace('-', '_')
		vector[ingredient2id[ingredient]] = ingredient2idf[ingredient]

	x = []
	for i in range(len(ingredient2idf)):
		if i in vector:
			x.append(vector[i])
		else:
			x.append(0)


	# id2vector[food['id']] = vector
	id2vector.append(x)
	# id2cuisine[food['id']] = food['cuisine']

	Y.append(cuisine2id[food['cuisine']])

print('now doing pca')
pca = PCA(n_components=60)
pca.fit(id2vector)
id2vector = pca.transform(id2vector)

# print('now doing metric learning')
# metricLearning = LMNN(k=3, learn_rate=1e-3, min_iter=3, max_iter=10)
# metricLearning.fit(id2vector, Y, verbose=False)

# id2vector = metricLearning.transform()

# print('finish metric learning')

test_data = []
with open('test.json') as test_file:
	testData = json.load(test_file)

	for data in testData:
		food = data

		vector = {}
		for j in xrange(len(food['ingredients'])):
			ingredient = food['ingredients'][j].lower().replace('-', '_')
			if ingredient in ingredient2id:
				vector[ingredient2id[ingredient]] = ingredient2idf[ingredient]

		x = []
		for i in range(len(ingredient2idf)):
			if i in vector:
				x.append(vector[i])
			else:
				x.append(0)

		test_data.append(x)


test_data = pca.transform(test_data)




# train = id2vector[:int(len(id2vector) * 0.7), :]
# test = id2vector[int(len(id2vector) * 0.7):, :]

# train_X = train
# train_Y = Y[:int(len(id2vector) * 0.7)]


# test_X = test
# test_Y = Y[int(len(id2vector) * 0.7):]

train_X = id2vector
train_Y = Y

test_X = test_data
test_Y = [0 for i in range(len(test_data))]


xg_train = xgb.DMatrix( train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)


param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 10
param['silent'] = 0
param['nthread'] = 8
param['num_class'] = 20
# clf = RandomForestClassifier(max_depth=35,n_estimators=300, n_jobs=20)
# clf.fit(id2vector, Y)

watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 5000
bst = xgb.train(param, xg_train, num_round, watchlist );
# get prediction


# test_X = test_data
# test_Y = [0 for i in range(len(test_data))]

# xg_test = xgb.DMatrix(test_X, label=test_Y)


prediction = bst.predict( xg_test );

# print ('predicting, classification error=%f' % (sum( int(pred[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))


# test_data = metricLearning.transform(test_data)

# prediction = clf.predict(test_data)


with open('test.json') as test_file:
	testData = json.load(test_file)
	with open('answer.csv', 'w') as w:
		w.write('id,cuisine\n')
		
		for i in xrange(len(testData)):
			food = testData[i]

			
			w.write(str(food['id']) + ',' + str(id2cuisine[prediction[i]]) + '\n')
			w.flush()


			


















