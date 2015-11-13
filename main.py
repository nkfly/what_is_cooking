import json
import math
import operator


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



	


with open('train.json') as data_file:
	trainData = json.load(data_file)


ingredient2documentCount = {}
ingredient2id = {}

for i in xrange(len(trainData)):
	food = trainData[i]

	for j in xrange(len(food['ingredients'])):
		if food['ingredients'][j] not in ingredient2id:
			ingredient2id[food['ingredients'][j]] = len(ingredient2id)

		if food['ingredients'][j] not in ingredient2documentCount:
			ingredient2documentCount[food['ingredients'][j]] = 1
		else:
			ingredient2documentCount[food['ingredients'][j]] += 1


# now idf is able to be counted
ingredient2idf = {}
for ingredient in ingredient2documentCount:
	ingredient2idf[ingredient] = math.log10(1+len(trainData)/ingredient2documentCount[ingredient])




id2vector = {}
id2cuisine = {}
for i in xrange(len(trainData)):
	food = trainData[i]

	vector = {}
	for j in xrange(len(food['ingredients'])):
		vector[ingredient2id[food['ingredients'][j]]] = ingredient2idf[food['ingredients'][j]]

	id2vector[food['id']] = vector
	id2cuisine[food['id']] = food['cuisine']


with open('test.json') as test_file:
	testData = json.load(test_file)


with open('answer.csv', 'w') as w:
	w.write('id,cuisine\n')
	
	for i in xrange(len(testData)):
		food = testData[i]

		testVector = {}

		for j in xrange(len(food['ingredients'])):
			if food['ingredients'][j] in ingredient2id:
				testVector[ingredient2id[food['ingredients'][j]]] = ingredient2idf[food['ingredients'][j]]


		id2cosineSimilarity = {}
		for trainId in id2vector:
			id2cosineSimilarity[trainId] = cosineSimilarity(id2vector[trainId], testVector)

		sortedId = sorted(id2cosineSimilarity.items(), key=operator.itemgetter(1), reverse=True)


		w.write(str(food['id']) + ',' + str(id2cuisine[sortedId[0][0]]) + '\n')
		w.flush()


			


















