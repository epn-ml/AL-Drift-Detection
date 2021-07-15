

#%% some imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#%% loading the labels

labels = pd.read_csv(r'labels(1-50).csv')

#%% setting datatype right


labels_date_list = ['SK outer in','SK inner in','MP outer in','MP inner in','MP inner out','MP outer out','SK inner out', 'SK outer out']

for date_feat in labels_date_list:
    labels[date_feat] = pd.to_datetime(labels[date_feat])

#%% loading the training data (50 orbits)

import glob

path = '\messenger-0000_-0050' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

df_train_init = pd.concat(li, axis=0, ignore_index=True)

df_train=df_train_init.dropna()

#%% change the datatype of the 'DATE' feature

df_train['DATE'] = pd.to_datetime(df_train['DATE'])

#%% transform the training data to a dataframe which contains the mean value over 1 minute

df_train_minute = df_train.copy()

df_train_minute.index = df_train_minute['DATE']

df_train_minute = df_train_minute.resample('1Min').mean()

df_train_minute = df_train_minute.reset_index()

#%% training data description

df_train_describtion = df_train.describe()

df_train_describtion.to_excel('df_train_description.xlsx')

df_train_minute_description = df_train_minute.describe()
df_train_minute_description.to_excel('df_train_minute_description.xlsx')

#%% assign labels to the instances of the training data

def labeller(df_train,labels):

    df_train['labels']='IMF'
    
    for i in range(0,len(labels)):
        
        # bow shock crossing
        df_train.loc[((df_train['DATE'] > labels.iloc[i]['SK outer in'])&(df_train['DATE'] < labels.iloc[i]['SK inner in'])) | 
                     ((df_train['DATE'] > labels.iloc[i]['SK inner out'])&(df_train['DATE'] < labels.iloc[i]['SK outer out']))
                     , 'labels'] = 'BS-crossing'  
        
        #magnetosheath
        df_train.loc[((df_train['DATE'] > labels.iloc[i]['SK inner in'])&(df_train['DATE'] < labels.iloc[i]['MP outer in'])) | 
                     ((df_train['DATE'] > labels.iloc[i]['MP outer out'])&(df_train['DATE'] < labels.iloc[i]['SK inner out'])) 
                     ,'labels'] = 'magnetosheath'  
        
        # magnetosphere    
        df_train.loc[(df_train['DATE'] > labels.iloc[i]['MP inner in'])&(df_train['DATE'] < labels.iloc[i]['MP inner out']), 'labels'] = 'magnetosphere'  
    
        # magnetopause crossing
        df_train.loc[((df_train['DATE'] > labels.iloc[i]['MP outer in'])&(df_train['DATE'] < labels.iloc[i]['MP inner in'])) | 
                     ((df_train['DATE'] > labels.iloc[i]['MP inner out'])&(df_train['DATE'] < labels.iloc[i]['MP outer out']))
                     , 'labels'] = 'MP-crossing'  
    return df_train
    
df_train_labelled = labeller(df_train,labels)
df_train_minute_labelled = labeller(df_train_minute,labels)
        
#%% df_train to_csv

df_train_labelled.to_csv('df_train_labelled.csv')
df_train_minute_labelled.to_csv('df_train_minute_labelled.csv') 
    
#%% describtion of the labelled training data
df_train_labelled = df_train_labelled.drop(['DATE'],axis=1)
df_train_labelled_description = df_train_labelled.groupby(['labels']).describe(include='all')
df_train_labelled_description.to_excel('df_train_labelled_description.xlsx')

df_train_minute_labelled = df_train_minute_labelled.drop(['DATE'],axis=1)
df_train_minute_labelled_description = df_train_minute_labelled.groupby(['labels']).describe(include='all')
df_train_minute_labelled_description.to_excel('df_train_minute_labelled_description.xlsx')
















