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
    st.title('당뇨병 예측')
    menu = ['home', 'EDA', 'ML']

    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'home':
        st.write('이 앱은 당뇨병 예측 앱입니다.')
    elif choice =='EDA':
        run_eda_app()
    elif choice == 'ML':
        run_ml_app()


if __name__ == '__main__':
    main()