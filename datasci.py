# Using xgboost algo
# in this code train and test are in one file so 
# we have to create separate code for test Pending...

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import nltk
import re
nltk.download('punkt')
from string import digits 
import heapq 
from xgboost import XGBClassifier

df = pd.read_csv('/root/Documents/datasci/Data/Final_Train_Dataset.csv')
df.head()
print(df.shape)

#experience
expri = df['experience']
expri = expri.str.split('-')
i=0
for year in expri:
    df.loc[i,'experience']= year[0]
    i+=1

#location
df.location.replace(pd.value_counts(df.location), inplace=True)  

#df.pop('Unnamed: 0')    #### popping index = do not do that

#filling na values
print(df.isnull().sum())
df.job_description.fillna('na', inplace=True)
df.job_type.fillna('na', inplace=True)
df.key_skills.fillna('na', inplace=True)
print(df.isnull().sum())

print(df.head())



#### Bag of Words ####

def bow(textdata):
    jobd = str(textdata)

    text = ''.join([i for i in jobd if not i.isdigit()]) 


#    print(len(text))
    dataset = nltk.word_tokenize(text)

#    print(dataset)
#    print(len(dataset))

    for i in range(len(dataset)):
        dataset[i] = dataset[i].lower()
        dataset[i] = re.sub(r'\W', ' ', dataset[i])
        dataset[i] = re.sub(r'\s+', ' ', dataset[i])

    word2count = {}
    for data in dataset:
        words = nltk.word_tokenize(data)
        for word in words:
            if word not in word2count.keys():
                word2count[word] = 1
            else:
                word2count[word] += 1

#    print(word2count)
#    print(len(word2count))    

#### wordlist ###
    freq_word = heapq.nlargest(1000, word2count, key=word2count.get)
    freq_word.sort(key= len, reverse=True)
    freq_word.append('na')

#    print(len(freq_word))
#    print(freq_word)

    ### iterate through each row ### 
    result = []
    for i in textdata:
        common = list(set([c for c in freq_word if c in i.lower()]))
        common.sort(key=len, reverse=True)
#        print(common)
        if not common:
            result.append(len(freq_word))
        else:
            result.append(freq_word.index(common[0]))

    
    return result       


##Function call

jobdescr = bow(df.job_description)
df.pop('job_description')
df['job_description'] = jobdescr

jobdesig = bow(df.job_desig)
df.pop('job_desig')
df['job_desig'] = jobdesig

jobtype = bow(df.job_type)
df.pop('job_type')
df['job_type'] = jobtype

keyskill = bow(df.key_skills)
df.pop('key_skills')
df['key_skills'] = keyskill

# change salary value
df.salary.replace({'0to3':0, '3to6':1, '6to10':2 , '10to15':3 , '15to25':4 ,'25to50':5}, 
                  inplace = True)
#seprate out result
salary = df.pop('salary')

#saving dataset
df.to_csv('/root/Documents/datasci/Data/savetrain.csv',index=False)
#trash = pd.read_csv('/root/Documents/ml/Data/savetrain.csv')
#df = trash

print(df.head())

# creating nparray 

x_train = df.to_numpy()
x_train = x_train.astype(np.int)
print(x_train.shape)

x_train, x_valid = x_train[:15841], x_train[15841:]



y_train = np.array(salary)
total_class = len(salary.unique())
print(y_train.shape)
y_train, y_valid = y_train[:15841], y_train[15841:]



# Create a classifier
model = XGBClassifier(booster='gbtree', objective='multi:softmax', 
                    learning_rate=0.1, eval_metric="auc", max_depth=14,
                    subsample=0.9, colsample_bytree=0.8, num_class=total_class,
                    n_estimators=500, gamma=5, alpha = 4)

# Fit the classifier with the training data

model.fit(x_train,y_train, verbose=True)


from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


predict = model.predict(x_train)

#create one hot encoding
lb = preprocessing.LabelBinarizer()

lb.fit(y_train)

y_train_lb = lb.transform(y_train)
val_lb = lb.transform(predict)

print(roc_auc_score(y_train_lb, val_lb, average='macro'))


print('train score')
print(accuracy_score(y_train, predict))

print('validation score')
val = model.predict(x_valid)

print(accuracy_score(y_valid, val))

output = pd.DataFrame()
output['Expected output'] = y_train
output['predicted output'] = predict
print(output)

output.to_csv('/root/Documents/datasci/Data/output.csv',index=False)