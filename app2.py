import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle
# from lightgbm import LGBMClassifier
import h5py
# from imblearn.over_sampling import SMOTE
from keras.models import load_model
from EDA_app import run_eda_app
from ML_app import run_ml_app


def main():
    model = joblib.load('data/best_model.pkl')
    df = pd.read_csv('data/diabetes.csv')
    st.dataframe(df)

    new_data = np.array([3,88,58,11,54,242,0,0.26,22])
    new_data = new_data.reshape(1,-1)
    print(new_data)

    st.write(model.predict(new_data))

    




if __name__ == '__main__':
main()