import numpy as np
import csv
#%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
word2vec=np.zeros((400001,51))
file=open("glove.6B.50d.txt","r")
counter=0
wordIndex={}
# Load the word vectors from another file
#Words vectors taken from https://nlp.stanford.edu/projects/glove/
# Trained using the Glove Algorithm
for line in file:
    counter=counter+1
    vec=line.split()
    word=vec[0]
    wordIndex[word]=counter
    for i in range(1,51):
        word2vec[counter][i]=float(vec[i])
# Load the word vectors from another file
parameter_size=[0 for j in range(1001)]
variables_size=[0 for j in range(1001)]
parameters=[[0 for j in range(51)] for i in range(1001)]
variables=[[0 for j in range(51)] for i in range(1001)]

counter=1
with open('variables.csv','r') as csvFile:
    reader=csv.reader(csvFile)
    for row in reader:
        # print(len(row))
        param=int(row[0])
        temp_variables=int(row[1])
        for j in range(2,2+param):
            parameter_size[counter]=parameter_size[counter]+1
            parameters[counter][j-1]=row[j]
        for j in range(param+2,param+2+temp_variables):
            variables_size[counter]=variables_size[counter]+1
            variables[counter][j-param-1]=row[j]
        counter=counter+1

def dist(a,b):
    vec_a=word2vec[a]
    vec_b=word2vec[b]
    mod_a=np.linalg.norm(vec_a)
    mod_b=np.linalg.norm(vec_b)
    return 1.0*np.dot(vec_a,vec_b)/(mod_a*mod_b)#/(1.0*np.absolute(vec_a)*np.absolute(vec_b))

similarity=[0 for i in range(1,1001)]
for i in range(1,1000):
    a=parameter_size[i]
    b=variables_size[i]
    count=0
    for j in range(1,a+1):
        for k in range(1,b+1):
            count=count+1
            if parameters[i][j]==variables[i][k]:
                similarity[i]=similarity[i]+1
                continue
            if parameters[i][j] in wordIndex and variables[i][k] in wordIndex:
                similarity[i]=similarity[i]+dist(wordIndex[parameters[i][j]],wordIndex[variables[i][k]])
    similarity[i]=similarity[i]*1.0/count
    if similarity[i]==1:
        print(i)

square_similarity=np.multiply(similarity,similarity)
x=np.linspace(1,1000,1000)
plt.plot(x,square_similarity,'o',color='black')
plt.show()
# print(similarity[2])
# print(similarity[3])