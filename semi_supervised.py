import json
import math
import operator
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import xgboost as xgb
import numpy as np
from sklearn.ensemble import VotingClassifier
import lda
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFECV
from metric_learn import LMNN
from metric_learn import SDML
import copy
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
id2ingredient = {}

cuisine2id = {}
id2cuisine = {}
ingredient2classCount = {}

stop_words = []

averageDocumentLength = 0.0

for i in xrange(len(trainData)):
	food = trainData[i]
	preprocessing_ingredients = []


	for j in xrange(len(food['ingredients'])):
		ingredients = food['ingredients'][j].lower().replace('-', ' ').split()
		for ingredient in ingredients:
			if ingredient in stop_words:
				continue
			if ingredient not in ingredient2id:
				ingredient2id[ingredient] = len(ingredient2id) # just give id
				id2ingredient[ingredient2id[ingredient]] = ingredient

			if ingredient not in ingredient2documentCount:
				ingredient2documentCount[ingredient] = 1
			else:
				ingredient2documentCount[ingredient] += 1

			if food['cuisine'] not in cuisine2id:
				cuisine2id[food['cuisine']] = len(cuisine2id)

				id2cuisine[cuisine2id[food['cuisine']]] = food['cuisine']


			if ingredient not in ingredient2classCount:
				ingredient2classCount[ingredient] = {food['cuisine'] : 1}
			else:
				ingredient2classCount[ingredient][food['cuisine']] = 1



			averageDocumentLength += 1


averageDocumentLength = averageDocumentLength / len(trainData)




# now idf is able to be counted
ingredient2idf = {}
for ingredient in ingredient2documentCount:
	ingredient2idf[ingredient] = math.log10(1+float(len(trainData))/ingredient2documentCount[ingredient]) 
	# ingredient2idf[ingredient] = 1




id2vector = []
# id2cuisine = {} # the trainId to cuisine(class)
Y = []
lda_X = []
for i in xrange(len(trainData)):
	food = trainData[i]

	vector = {}
	for j in xrange(len(food['ingredients'])):
		ingredients = food['ingredients'][j].lower().replace('-', ' ').split()
		vector['length'] = 0
		for ingredient in ingredients:
			if ingredient in stop_words:
				continue
			# if ingredient2id[ingredient] not in vector:
			# 	vector[ingredient2id[ingredient]] = ingredient2idf[ingredient]
			# else:
			# 	vector[ingredient2id[ingredient]] += ingredient2idf[ingredient]
			vector['length'] += 1

			if ingredient2id[ingredient] not in vector:
				vector[ingredient2id[ingredient]] = 1
			else:
				vector[ingredient2id[ingredient]] += 1




	x = []
	lda_x = []
	for i in range(len(ingredient2idf)):
		if i in vector:
			x.append(ingredient2idf[id2ingredient[i]])
			lda_x.append(vector[i])

		else:
			x.append(0)
			lda_x.append(0)

	id2vector.append(x)
	lda_X.append(lda_x)	

	Y.append(cuisine2id[food['cuisine']])


# lda
model = lda.LDA(n_topics=20, n_iter=500, random_state=1)
model.fit(np.array(lda_X))  # model.fit_transform(X) is also available
topic_word = model.topic_word_  # model.components_ also works

print topic_word

for i in range(len(id2vector)):
	topic_distribution = [0 for t in range(20)]

	for j in range(len(id2vector[i])):
		if id2vector[i][j] == 0:
			continue


		max_k = 0
		max_proba = 0.0
		for k, topic_dist in enumerate(topic_word):
			#topic_distribution[k] += topic_dist[j]
			if topic_dist[j] > max_proba:
				max_proba = topic_dist[j]
				max_k = k


		topic_distribution[max_k] += 1



	id2vector[i].extend(topic_distribution)

print len(id2vector[0])

test_data = []
with open('test.json') as test_file:
	testData = json.load(test_file)

	for data in testData:
		food = data

		vector = {}
		vector['length'] = 0
		for j in xrange(len(food['ingredients'])):
			ingredients = food['ingredients'][j].lower().replace('-', ' ').split()
			for ingredient in ingredients:
				if ingredient in stop_words:
					continue
				vector['length'] += 1
				if ingredient in ingredient2id:
					if ingredient2id[ingredient] not in vector:
						vector[ingredient2id[ingredient]] = ingredient2idf[ingredient]
					# else:
					# 	vector[ingredient2id[ingredient]] += ingredient2idf[ingredient]

		x = []
		for i in range(len(ingredient2idf)):
			if i in vector:
				x.append(vector[i])
			else:
				x.append(0)
		#x.append(vector['length'])
		test_data.append(x)


for i in range(len(test_data)):
	topic_distribution = [0 for t in range(20)]

	for j in range(len(test_data[i])):
		if test_data[i][j] == 0:
			continue


		max_k = 0
		max_proba = 0.0
		for k, topic_dist in enumerate(topic_word):
			#topic_distribution[k] += topic_dist[j]
			
			if topic_dist[j] > max_proba:
				max_proba = topic_dist[j]
				max_k = k


		topic_distribution[max_k] += 1


	test_data[i].extend(topic_distribution)

train_X = id2vector
train_Y = Y

test_X = test_data
test_Y = [0 for i in range(len(test_data))]


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
param['num_class'] = 20

watchlist = [ (xg_train,'train')]
num_round = 800
bst = xgb.train(param, xg_train, num_round, watchlist)

prediction = bst.predict( xg_test )

_test_X = test_data

for iteration in range(5):
	print 'len of training is ' + str(len(id2vector))
	maxProbArray = []
	for i in range(len(prediction)):
		maxProbArray.append(np.amax(prediction[i]))

	maxProbArray.sort()
	new_test_X = []

	for i in range(len(prediction)):
		if np.amax(prediction[i]) > maxProbArray[int(len(maxProbArray)/2)]:
			id2vector.append(_test_X[i])
			Y.append(np.argmax(prediction[i]))
		else:
			new_test_X.append(_test_X[i])



	train_X = id2vector
	train_Y = Y

	test_X = new_test_X
	test_Y = [0 for i in range(len(new_test_X))]


	xg_train = xgb.DMatrix(train_X, label=train_Y)
	xg_test = xgb.DMatrix(test_X, label=test_Y)

	bst = xgb.train(param, xg_train, num_round, watchlist)
	prediction = bst.predict( xg_test )

	_test_X = new_test_X


train_X = id2vector
train_Y = Y

test_X = test_data
test_Y = [0 for i in range(len(test_data))]

xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)

param['objective'] = 'multi:softmax'

bst = xgb.train(param, xg_train, num_round, watchlist)

prediction = bst.predict( xg_test )


with open('test.json') as test_file:
	testData = json.load(test_file)
	with open('answer.csv', 'w') as w:
		w.write('id,cuisine\n')
		
		for i in xrange(len(testData)):
			food = testData[i]

			
			w.write(str(food['id']) + ',' + str(id2cuisine[prediction[i]]) + '\n')
			w.flush()


			


















