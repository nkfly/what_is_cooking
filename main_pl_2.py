import json
import math
import operator
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier


""" 
load train data
"""
def json2dict(filename):
	with open(filename) as data_file:
		thedict =  json.load(data_file)
	return thedict



"""
Training data to sparse array:
"""
def train_dict2array(filename):
	answer = []
	data = []
	row = []
	col = []
	count = 0
	source = json2dict(filename)
	cousine = json2dict("cooking_data/id2cuisine.json")
	for k1, v1 in source.items():
		r = count
		for k2, v2 in v1.items():
			c = int(k2)
			data.append(1)
			row.append(r)
			col.append(c)
		count += 1
		answer.append(cousine[k1])
	# Create the COO-matrix
	coo = coo_matrix((data,(row,col)))
	# Let Scipy convert COO to CSR format and return
	return csr_matrix(coo), answer



"""
Testing data to sparse array:
"""
def test_dict2array(filename, dimension):
	id = []
	data = []
	row = []
	col = []
	count = 0
	source = json2dict(filename)
	for k1, v1 in source.items():
		r = count
		for k2, v2 in v1.items():
			c = int(k2)
			data.append(1)
			row.append(r)
			col.append(c)
		id.append(k1)
		count += 1
	# Create the COO-matrix
	coo = coo_matrix((data,(row,col)), shape=(len(source), dimension))
	# Let Scipy convert COO to CSR format and return
	return csr_matrix(coo), id



'''
"""
Prepare testing set:

e.g., testVector = {ing_id: inf, ing_id_2: inf, ...}
"""
with open('test.json') as test_file:
	testData = json.load(test_file)

with open('answer.csv', 'w') as w:
	w.write('id,cuisine\n')
	
	for i in xrange(len(testData)):
		food = testData[i]

		testVector = {}

		# make a test vector
		for j in xrange(len(food['ingredients'])):
			if food['ingredients'][j] in ingredient2id:
				testVector[ingredient2id[food['ingredients'][j]]] = ingredient2idf[food['ingredients'][j]]

		id2cosineSimilarity = {}
		for trainId in id2vector:
			# calculate the test vector cosine similarity with every vector in train
			id2cosineSimilarity[trainId] = cosineSimilarity(id2vector[trainId], testVector)

		# sort to find the most similar vector in train
		sortedId = sorted(id2cosineSimilarity.items(), key=operator.itemgetter(1), reverse=True)

		# the test cuisine is the most similar train vector cuison
		w.write(str(food['id']) + ',' + str(id2cuisine[sortedId[0][0]]) + '\n')
		w.flush()
'''

def main():
	'''
	#check properties
	id2vector = json2dict('cooking_data/id2vector.json')
	print len(id2vector)
	keys = [ int(k) for k in id2vector.keys()]
	print min(keys), max(keys)
	'''

	train, answer = train_dict2array('cooking_data/id2vector.json')
	test, id = test_dict2array('cooking_data/testVector.json', train.shape[1])
	print train.shape, len(answer), test.shape
	print answer[0:10]
	#clf = AdaBoostClassifier()
	clf = RandomForestClassifier()
	clf.fit(train, answer)
	result = clf.predict(test)
	print result[0:10]

	with open('answer_pl_2.csv', 'w') as w:
		w.write('id,cuisine\n')
		for i in xrange(len(result)):
			w.write(id[i] + ',' + result[i] + '\n')
			w.flush()

if __name__ == '__main__':
    main()















