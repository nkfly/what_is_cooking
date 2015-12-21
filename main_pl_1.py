import json
import math
import operator


"""
dump dict to json test_file
"""
def dict2json(directory, filename, thedict):
	j = json.dumps(thedict, indent=4)
	if directory is not None:
		directory += '/'
	else:
		directory = ''
	f = open(directory + filename + '.json', 'w')
	print >> f, j
	f.close()



""" 
load train data
"""
with open('train.json') as data_file:
	trainData = json.load(data_file)



"""
Make basic dictionary:

e.g., ingredient2documentCount = {'pasta': 2} means pasta appears in 2 receipts
e.g., ingredient2id = {'pasta', 1} means the ID of pasta is 1
"""
ingredient2documentCount = {} 
ingredient2id = {} 

for i in xrange(len(trainData)):
	food = trainData[i]
	for j in xrange(len(food['ingredients'])):
		if food['ingredients'][j] not in ingredient2id:
			ingredient2id[food['ingredients'][j]] = len(ingredient2id) # just give id
		if food['ingredients'][j] not in ingredient2documentCount:
			ingredient2documentCount[food['ingredients'][j]] = 1
		else:
			ingredient2documentCount[food['ingredients'][j]] += 1

dict2json('cooking_data', 'ingredient2documentCount', ingredient2documentCount)
dict2json('cooking_data', 'ingredient2id', ingredient2id)

# now idf is able to be counted
ingredient2idf = {}
for ingredient in ingredient2documentCount:
	ingredient2idf[ingredient] = math.log10(1+len(trainData)/ingredient2documentCount[ingredient])

dict2json('cooking_data', 'ingredient2idf', ingredient2idf)



"""
Prepare training set:

e.g., id2vector = {food_id: {ing_id: inf, ing_id_2: inf, ...} }
e.g., id2cuisine = {'food_id': 'cuisine_name'}
				= {'2': 'sushi'} 
"""
id2vector = {} # the trainId to vector, the dimension in vector is ingredient, value is the idf
id2cuisine = {} # the trainId to cuisine(class)
for i in xrange(len(trainData)):
	food = trainData[i]

	vector = {}
	for j in xrange(len(food['ingredients'])):
		vector[ingredient2id[food['ingredients'][j]]] = ingredient2idf[food['ingredients'][j]]

	id2vector[food['id']] = vector
	id2cuisine[food['id']] = food['cuisine']

dict2json('cooking_data', 'id2vector', id2vector)
dict2json('cooking_data', 'id2cuisine', id2cuisine)



"""
Prepare testing set:

e.g., testVector = {ing_id: inf, ing_id_2: inf, ...}
"""
with open('test.json') as test_file:
	testData = json.load(test_file)

testVector = {}
for i in xrange(len(testData)):
	food = testData[i]

	vector = {}
	for j in xrange(len(food['ingredients'])):
		if food['ingredients'][j] in ingredient2id:
			vector[ingredient2id[food['ingredients'][j]]] = ingredient2idf[food['ingredients'][j]]
	testVector[food['id']] = vector

dict2json('cooking_data', 'testVector', testVector)




