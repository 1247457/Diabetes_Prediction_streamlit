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
# from EDA_app import run_eda_app
# from ML_app import run_ml_app


def run_ml_app() :
    st.subheader('머신 러닝 화면입니다.')


    # 1. 유저에게 입력을 받는다. 

    preg = st.number_input('임신횟수', min_value=0, max_value=120)

    glguc = st.number_input('공복혈당', min_value=0)

    bldprs = st.number_input('혈압', min_value=0)

    insulin = st.number_input('인슐린', min_value=0)

    bmi = st.number_input('BMI', min_value=0)

    age = st.number_input('나이', min_value=0)




