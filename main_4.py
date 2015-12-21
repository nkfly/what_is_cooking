import json
import math
import operator
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from metric_learn import LMNN
from scipy.spatial.distance import cosine
import xgboost as xgb
import numpy as np

	

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
pca = PCA(n_components=100)
pca.fit(id2vector)
id2vector = pca.transform(id2vector)



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


prediction = []
for t_data in test_data:
	id2value = {}
	for i in range(len(id2vector)):
		train_data = id2vector[i]	
		cosineValue = cosine(train_data, t_data)
		id2value[i] = cosineValue

	coo = sorted(id2value.items(), key=operator.itemgetter(1))
	prediction.append(Y[coo[0][0]])








with open('test.json') as test_file:
	testData = json.load(test_file)
	with open('answer4.csv', 'w') as w:
		w.write('id,cuisine\n')
		
		for i in xrange(len(testData)):
			food = testData[i]

			
			w.write(str(food['id']) + ',' + str(id2cuisine[prediction[i]]) + '\n')
			w.flush()


			


















