import numpy as np
import pandas as pd
from os.path import join
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor

def GetData():
    df_tr = GetTrainingDf()
    df_te = GetTestDf()
    size_training = len(df_tr)
   
    y_tr = df_tr['y'].values
    df_tr.drop(['y'], axis=1, inplace=True)
    df = pd.concat([df_tr, df_te], ignore_index=True)
    df_string = df.select_dtypes(exclude=['number'])
    df_bool = df.select_dtypes(include=['number'])
    df_onehot = pd.get_dummies(df_string)
    X = np.hstack([df_onehot.values, df_bool.values])
    X_tr = X[:size_training, :]
    X_te = X[size_training:, :]
    return X_tr, y_tr, X_te
    

def GetTestID():
    df = GetTestDf()
    return df['ID']
def GetTrainingDf():
    path = join('data', 'train.csv')
    df = pd.read_csv(path)
    return df
    
def GetTestDf():
    path = join('data', 'test.csv')
    df = pd.read_csv(path)
    return df
    
def GenerateSubmissionFile(model, X):
    pred = model.predict(X)
    ids = GetTestID()
    df_result = pd.DataFrame()
    df_result['ID'] = ids
    df_result['y'] = pred
    path = join('data', 'submission.csv')
    df_result.to_csv(path, sep=',', index =False)
    
def Split(X, y, train_size=0.9):
    if train_size==1:
        return X, None, y, None
        
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, train_size=train_size)
    return X_tr, X_val, y_tr, y_val
    
if __name__=='__main__':
    
    X_tr, y_tr, X_te = GetData()
    X_tr, X_val, y_tr, y_val = Split(X_tr, y_tr, train_size=1)
    reg = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=4, verbose=1)
    reg.fit(X_tr, y_tr)
    
    if X_val is not None and y_val is not None:
        score = reg.score(X_val, y_val)
        print('our score is {}'.format(score))
        
    GenerateSubmissionFile(reg, X_te)
        
    
        
        
    
    
    
    
    
    
    

