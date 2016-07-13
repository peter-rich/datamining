# -*- coding: utf-8 -*-
######################################################################
###    according to Internet's help I write this program which do not use already package to 
####    run logistic_regression
########################################################################
import json
import numpy as np
import csv


t_set = set()
x_set = set() #x : feature

learning_rate = 1

alpha = 0.0002 #惩罚系数
iter_time = 3

if __name__ == "__main__":
    with open("train.json") as json_file:
        train_data = json.load(json_file) #list
        N = len(train_data) #number of training data
        for datai in train_data:
            t_set.add(datai['cuisine'])
            for ingredient in datai['ingredients']:
                x_set.add(ingredient)



    K = len(t_set)
    M = len(x_set)
    t_list = list(t_set)
    x_list = list(x_set)

    T = np.zeros([N,K])
   
    X = np.zeros([N,M])
    for i in range(N):
        datai = train_data[i]
        cuisine_i = datai['cuisine']
        cuisine_i_index = t_list.index(cuisine_i)
        T[i,cuisine_i_index] = 1
        x_i = datai['ingredients']
        for itemj in x_i:
            itemj_index = x_list.index(itemj)
            X[i,itemj_index] = 1



    pass
    
    W = np.zeros([K,M])
   
    for i in range(K):
        for j in range(M):
            W[i,j] = np.random.rand(1) -0.5

    
    training_label = np.zeros(N)
    for i in range(N):
        if np.random.rand(1)<1.01: 
            training_label[i] = 1

    
    y = np.zeros(K) #prediction y
    for iter in range(iter_time): 
        learning_rate = learning_rate*0.1
        print "training: ",iter," time"
        for i in range(N): 
            if training_label[i] == 1:
                summ = 0.0
                for j in range(K): 
                    y[j] = np.exp( np.dot(W[j],X[i]) )
                    summ += y[j]
                for j in range(K): 
                    y[j] = y[j]/summ

                for j in range(K):
                    W[j] = W[j] - learning_rate*(y[j]-T[i,j])*X[i] - alpha * learning_rate * W[j]


    print "Finish training"



    print "begin testing"
    with open("test.json") as json_file:
        test_data = json.load(json_file) #list
        N_test = len(test_data)


    
    with open('test_prediction.csv', 'wb') as csvfile:
        fieldnames = ['id' , 'cuisine']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        X_test = np.zeros([N_test,M])
        for i in range(N_test):
            datai = test_data[i]
            x_i = datai['ingredients']
            id = datai['id']
            for itemj in x_i:
                if itemj in x_list:
                    itemj_index = x_list.index(itemj)
                    X_test[i,itemj_index] = 1

         
            for j in range(K):
                y[j] = np.exp( np.dot(W[j],X_test[i]) )
            max_index = y.argmax(axis=0)
            cuisine = t_list[max_index]
            writer.writerow({'id': id, 'cuisine': cuisine})

