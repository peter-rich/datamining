###########################################################
# To find out the country and cuisine information from
# the .json file
###########################################################


import pandas as pd
import numpy as np
import json
import csv
import tempfile
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.grid_search import GridSearchCV



import logging

trainf = pd.read_json('train.json')
testf = pd.read_json('test.json')
train_x =[]
test_x =[]
import json
import csv
import tempfile
f = open('train.json')
data = json.load(f)
dat= json.dumps(data)
dats = json.loads(dat)
#print(data[0])
f.close()

judge = 0;



total_country = ['southern_us']
total_country_number = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
total_country_material=[[]for i in range(20)]
total_country_material[0].append('plain flour')
total_country_material_number=[[]for i in range(20)]
total_country_material_number[0].append(0)

for j in range(0,len(dats)):
	true = 0;
	i = 0;
	for i in range(0,len(total_country)):
		if(total_country[i]==dats[j]['cuisine']):
			true = 1;
			total_country_number[i]+=1;
			break;
								
	if(true == 0):
		total_country.append(dats[j]['cuisine']);
		total_country_number[len(total_country)-1]+=1;
		total_country_material[len(total_country)-1].append(dats[j]['ingredients'][0]);
		total_country_material_number[i].append(1);
				

print(total_country_material);
print(total_country);
print(total_country_number);
print(len(total_country_number));
total=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
print(trainf['ingredients'][0]);
cuisine = []
all_sum = len(total_country_number)
"""
for i in range(0,20):
	all_sum += total[i];

print(all_sum); 
for i in range(0,20):
	total[i]/=all_sum;
"""
for i in range(0,len(dats)):
	for j in range(0,20):
		if((total[j]<1000 or total[j]<total_country_number[j]*0.8)and total_country[j]==trainf['cuisine'][i]):
			total[j]+=1;
			train_x.append(' '.join(trainf['ingredients'][i]))
			cuisine.append(trainf['cuisine'][i]);
"""
for i in range(0,len(dats)):
	print(i)
	for j in range(0,20):
		if(total[j]<6200 and total_country[j]==trainf['cuisine'][i]):			
			total[j]+=1;
			#print(trainf['ingredients'][i])
			trainf_i = []
			trainf_c =[]
			for k in range(0,len(trainf['ingredients'][i])):
				ch = []
				for p in range(len(trainf['ingredients'][i][k])-1,-1,-1):
					if(p==0):
						for q in range(p,len(trainf['ingredients'][i][k])):
							if(trainf['ingredients'][i][k][q]!=' '):
								ch.append(trainf['ingredients'][i][k][q]);
						break;
				
				ch_now = "".join(ch);
				#print(ch_now);
				trainf_c.append(ch_now)
				#print(trainf['ingredients'][i][k])
				trainf['ingredients'][i][k]=ch_now
				#print(trainf['ingredients'][i][k])
				
			train_x.append(' '.join(trainf['ingredients'][i]))
			
			cuisine.append(trainf['cuisine'][i]);
#print(train_x)
"""
for x in testf['ingredients']:
	test_x.append(' '.join(x))
"""
f = open('test.json')
data = json.load(f)
dat= json.dumps(data)
dats = json.loads(dat)
#print(data[0])
f.close()
for i in range(0,len(dats)):
		print(i)
		#print(trainf['ingredients'][i])
		testf_i = []
		testf_c =[]
		for k in range(0,len(testf['ingredients'][i])):
			ch = []
			for p in range(len(testf['ingredients'][i][k])-1,-1,-1):
				if(p==0):
					for q in range(p,len(testf['ingredients'][i][k])):
						if(testf['ingredients'][i][k][q]!=' '):
							ch.append(testf['ingredients'][i][k][q]);
					break;
				
			ch_now = "".join(ch);
			#print(ch_now);
			testf_c.append(ch_now)
			#print(testf['ingredients'][i][k])
			testf['ingredients'][i][k]=ch_now
			#print(testf['ingredients'][i][k])
			
		test_x.append(' '.join(testf['ingredients'][i]))
"""		
print(test_x)


vectorizer = TfidfVectorizer()
train_x = vectorizer.fit_transform(train_x)
test_x = vectorizer.transform(test_x)

#print(train_x)
#print(test_x)
tuned_parameters = [
{'penalty':['l1'],'tol':[1e-3,1e-4],'C':[1,10,100,1000]},{'penalty':['l2'],'tol':[1e-3,1e-4],
'C':[1,10,100,1000]}]
fun=LinearSVC()
clf = GridSearchCV(fun,tuned_parameters,cv=5,scoring=['precision','recall'])
print(clf)
#LogisticRegression(C = 1.0, intercept_scaling=5,dual=False,fit_intercept=True,penalty='l2',tol=0.00001)
fun.fit(train_x,cuisine)

results = fun.predict(test_x)
testf['results'] = results
testf.to_csv('final.csv')

