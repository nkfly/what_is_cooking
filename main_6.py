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


averageDocumentLength = 0.0

for i in xrange(len(trainData)):
	food = trainData[i]
	preprocessing_ingredients = []


	for j in xrange(len(food['ingredients'])):
		ingredients = food['ingredients'][j].lower().replace('-', ' ').split()
		for ingredient in ingredients:
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
	# ingredient2idf[ingredient] = math.log10(1+20.0/len(ingredient2classCount[ingredient])) 







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


	# id2vector[food['id']] = vector
	id2vector.append(x)
	lda_X.append(lda_x)
	# id2cuisine[food['id']] = food['cuisine']

	Y.append(cuisine2id[food['cuisine']])


# lda
model = lda.LDA(n_topics=20, n_iter=500, random_state=1)
print np.array(lda_X)
model.fit(np.array(lda_X))  # model.fit_transform(X) is also available
topic_word = model.topic_word_  # model.components_ also works

for i in range(len(id2vector)):
	topic_distribution = [0 for t in range(20)]

	for j in range(len(id2vector[i])):
		if id2vector[i][j] == 0:
			continue


		max_k = 0
		max_proba = 0.0
		for k, topic_dist in enumerate(topic_word):
			if topic_dist[k] > max_proba:
				max_proba = topic_dist[k]
				max_k = k


		topic_distribution[max_k] += 1



	id2vector[i].extend(topic_distribution)


clf = RandomForestClassifier(max_depth=35,n_estimators=500, n_jobs=20)
clf.fit(id2vector, Y)

y_pred = clf.predict(id2vector)

for i in range(len(y_pred)):
	id2vector[i].append(y_pred[i])
	

print len(id2vector[0])

			







# print('now doing pca')
# pca = PCA(n_components=100)
# pca.fit(id2vector)
# id2vector = pca.transform(id2vector)

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
			ingredients = food['ingredients'][j].lower().replace('-', ' ').split()
			for ingredient in ingredients:
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

		test_data.append(x)


for i in range(len(test_data)):
	topic_distribution = [0 for t in range(20)]

	for j in range(len(test_data[i])):
		if test_data[i][j] == 0:
			continue


		max_k = 0
		max_proba = 0.0
		for k, topic_dist in enumerate(topic_word):
			if topic_dist[k] > max_proba:
				max_proba = topic_dist[k]
				max_k = k


		topic_distribution[max_k] += 1



	test_data[i].extend(topic_distribution)

# test_data = pca.transform(test_data)

y_pred = clf.predict(test_data)

for i in range(len(y_pred)):
	test_data[i].append(y_pred[i])




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
num_round = 800
bst = xgb.train(param, xg_train, num_round, watchlist );
# get prediction

# eclf = VotingClassifier(estimators=[('bst', bst)], voting='soft')
# eclf.fit(id2vector, Y)
# test_X = test_data
# test_Y = [0 for i in range(len(test_data))]

# xg_test = xgb.DMatrix(test_X, label=test_Y)

# prediction = eclf.predict(xg_test)
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


			


















