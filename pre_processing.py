import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns









def target_encode(total,train,lab):
    total[lab]=total[lab].map(train.groupby(train[lab])['is_promoted'].mean())
    
    
    
def label_encode(lab,data,test=pd.DataFrame(),test_provided=False,return_encoder=False):
    un = np.array(data[lab].unique())
    un = un[pd.notnull(un)]
    mapping = dict(zip(un,range(0,len(un))))
    data[lab] = data[lab].map(mapping).values
    if(test_provided==True):
        test[lab] = test[lab].map(mapping).values
    if(return_encoder==True):
        return(le)
    
    
    
def impute(df,lab):
    print('imputing:',lab)
    lb_train=df[df[lab].notnull()]
    lb_test=df[df[lab].isnull()]
    for c in lb_train.columns[lb_train.columns != lab]:
        if(lb_train[c].dtype=='object'):
            label_encode(c,lb_train,lb_test,test_provided=True,return_encoder=False)
            lb_train[c] = pd.to_numeric(lb_train[c])
    y_lb=lb_train[lab].values
    xtr = lb_train.drop(lab,axis=1).values
    xte = lb_test.drop(lab,axis=1).values
    from xgboost import XGBRegressor as XGR,XGBClassifier as XGC
    if(len(df[lab].unique())<50):
      xgb=XGC()
      print("  It is categorical")
    else:
      xgb=XGR()
      print("  It is numerical")
    xgb.fit(xtr,y_lb)
    yte=(xgb.predict(xte))
    df[lab][df[lab].isnull()]=yte