# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json
import csv
import tempfile
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.grid_search import GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import json
import csv
import pandas as pd
import sys 
reload(sys) 
sys.setdefaultencoding('utf8') 
sys.setdefaultencoding('gb18030')

t_set = set()
x_set = set()
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

if __name__ == "__main__":  
    with open('train.json') as json_file:
	train_data = json.load(json_file) 
        N = len(train_data)  
	for datai in train_data:
		t_set.add(datai['cuisine']) 
		for ingredient in datai['ingredients']:
			x_set.add(ingredient)   
	traindf = pd.read_json('train.json')	
	K = len(t_set)    
	M = len(x_set)    
	t_list = list(t_set)  
	x_list = list(x_set)
	
	T = np.zeros(N) 
        targets_tr = traindf['cuisine']
	X = np.zeros([N, M])
	
	print "start train"
        for i in range(N):
            datai = train_data[i]
            cuisine_i = datai['cuisine']
            cuisine_i_index = t_list.index(cuisine_i) 
            T[i] = cuisine_i_index 
            x_i = datai['ingredients']
            for itemj in x_i:
                itemj_index = x_list.index(itemj)
                X[i, itemj_index] = 1
         
        eclf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
        eclf = eclf.fit(X,T)
        
        
        print "end training"
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
        print "start testing"
        with open('test.json') as json_file:
            test_data = json.load(json_file) 
            N_test = len(test_data)
            
        with open('test_prediction(rf).csv','wb') as csvfile:
            fieldnames = ['id','cuisine']
            writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
            writer.writeheader()
            
            X_test = np.zeros([N_test, M])
            for i in range(N_test):
                datai = test_data[i]
                x_i = datai['ingredients']
                id = datai['id']
                for itemj in x_i:
                    if itemj in x_list:
                        itemj_index = x_list.index(itemj)
                        X_test[i, itemj_index] = 1
                        
           
            predict_target = eclf.predict(X_test)
            for i in range(N_test):
                datai = test_data[i]
                x_i = datai['ingredients']
                id = datai['id']
                cuisine = t_list[int(predict_target[i])]
                writer.writerow({'id':id, 'cuisine':cuisine})
                
        
