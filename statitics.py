import json

with open('test.json') as test_file:
	testData = json.load(test_file)

print len(testData)