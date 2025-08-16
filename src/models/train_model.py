import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from numpy.typing import NDArray
import pickle
import logging
import os

# logging.basicConfig()

def load_dataset(path:str)-> tuple[NDArray[np.float64],NDArray[np.float64]]:
    try:
        train_tfidf=pd.read_csv(path)
        # print(train_tfidf.shape)
        x_train=train_tfidf.drop("target",axis=1).values
        y_train=train_tfidf["target"].values
        return x_train,y_train
    except FileNotFoundError as e:
        print(e)
        raise
def train_model(x_train:NDArray,y_train:NDArray)->RandomForestClassifier:
    rf=RandomForestClassifier(n_estimators=500,max_leaf_nodes=100,criterion='gini',max_depth=17)
    rf.fit(x_train,y_train)
    return rf

def save_model(path:str,model_obj:RandomForestClassifier)->None:
    try:
     os.makedirs(os.path.dirname(path),exist_ok=True)
     with open(path,"wb") as f:
        pickle.dump(model_obj,f)
    except FileNotFoundError as e:
       print(e)
       raise

def main():
   x_train,y_train=load_dataset("data/processed/train_tfidf.csv")
   rf=train_model(x_train,y_train)
   save_model("models/model.pkl",rf)

if __name__=="__main__" :
   main()