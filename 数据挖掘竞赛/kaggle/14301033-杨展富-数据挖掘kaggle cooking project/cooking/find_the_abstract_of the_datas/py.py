#################################################
# logRegression: Logistic Regression Grid_search
#################################################

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
	print(j);
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
				

#print(total_country_material);
#print(total_country);
#print(total_country_number);
#print(len(total_country_number));

all_material=[dats[j]['ingredients'][0]]
all_total=[0]
for j in range(len(dats)/100,len(dats)/50):# control the sampling. 
    print(j)
    dats[j]['id']
    dats[j]['cuisine']
    for i in range(0,len(dats[j]['ingredients'])):
        for k in range(0,len(all_material)):
            if(dats[j]['ingredients'][i] == all_material[k]):judge = 1;all_total[k]+=1;break;
        if(judge==1):
            judge = 0;
        else:
            all_material.append(dats[j]['ingredients'][i]);
            all_total.append(0);

index =[]

for j in range(0,len(all_total)):
    if( all_total[j]>8 and all_total[j]<11): #control the times it appears. 
        index.append(j);

rule_name = []

for i in range(0,len(index)):
    rule_name.append(all_material[index[i]]);

dats_value = []#
for j in range(0,len(dats)):
    values = []
    
    dats[j]['id']
    dats[j]['cuisine']
    for i in range(0,len(index)):
        value=0
        for k in range(0,len(dats[j]['ingredients'])):
            if(dats[j]['ingredients'][k]==rule_name[i]):value+=1;
        values.append(value);
    values.append(dats[j]['id']);
    values.append(dats[j]['cuisine']);

    dats_value.append(values);


print(dats_value[1]);
print(len(all_material));
print(len(index));
f.close()


#def write_file(path, datas):
#    with open(path, 'w') as cf:
 #       writer = csv.writer(cf, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
#        for row in datas:
#           writer.writerow(row)
#
#def write_files(path, datas):
#    f = open(path,'w')
#    for i in range(0,len(datas)):
#        f.write(datas[i]);
#        f.write('\n')
#    f.close()

#def main():
#    init()

#datas = [['id','cuisine'],['001', 'China'],['002', 'American'],['003', 'Italy'],['004', 'West']]
print('#' * 50)
#write_file('class2.csv',  dats_value)
#write_files('train_features_file.txt',  rule_name)

#    if __name__ == '__main__':
#        main()
# -*- coding: utf-8 -*-

