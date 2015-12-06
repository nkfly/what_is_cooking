import json
import math
import operator
from sklearn.ensemble import RandomForestClassifier

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
		if food['ingredients'][j] not in ingredient2id:
			ingredient2id[food['ingredients'][j]] = len(ingredient2id) # just give id

		if food['ingredients'][j] not in ingredient2documentCount:
			ingredient2documentCount[food['ingredients'][j]] = 1
		else:
			ingredient2documentCount[food['ingredients'][j]] += 1

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
		vector[ingredient2id[food['ingredients'][j]]] = ingredient2idf[food['ingredients'][j]]

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

clf = RandomForestClassifier(max_depth=30,n_estimators=200, n_jobs=20)
clf.fit(id2vector, Y)


test_data = []
with open('test.json') as test_file:
	testData = json.load(test_file)

	for data in testData:
		food = data

		vector = {}
		for j in xrange(len(food['ingredients'])):
			if food['ingredients'][j] in ingredient2id:
				vector[ingredient2id[food['ingredients'][j]]] = ingredient2idf[food['ingredients'][j]]

		x = []
		for i in range(len(ingredient2idf)):
			if i in vector:
				x.append(vector[i])
			else:
				x.append(0)

		test_data.append(x)




prediction = clf.predict(test_data)
with open('test.json') as test_file:
	testData = json.load(test_file)
	with open('answer.csv', 'w') as w:
		w.write('id,cuisine\n')
		
		for i in xrange(len(testData)):
			food = testData[i]

			
			w.write(str(food['id']) + ',' + str(id2cuisine[prediction[i]]) + '\n')
			w.flush()


			


















