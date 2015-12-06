import json

with open('train.json') as test_file:
	testData = json.load(test_file)
	count = {}
	for data in testData:
		# print data['cuisine']

		for ingredient in data['ingredients']:
			count[ingredient] = 1

	print len(count)




# print len(testData)