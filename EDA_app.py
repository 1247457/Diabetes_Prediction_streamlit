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



def run_eda_app() :
    st.subheader('EDA 화면입니다.')

    # 1. 데이터프레임 보기
    df = pd.read_csv('data/diabetes.csv', encoding='ISO-8859-1')
    st.dataframe(df)



    # 2. 컬럼별 보기
    columns = df.columns
    columns = list(columns)
    multi = st.multiselect('확인할 컬럼을 선택해주세요', columns)
    st.dataframe(df[multi])



    # 2. 상관관계 보기
    corr_columns = df.columns[ df.dtypes != object ]
    selected_corr = st.multiselect('상관계수 컬럼 선택', corr_columns)

    if len(selected_corr) > 0 :
        st.dataframe(df[selected_corr].corr())
        # 위의 선택한 컬럼들을 이용해 씨본의 페어플롯을 그린다.
        fig = sns.pairplot( data = df[selected_corr] )
        st.pyplot(fig)
    else :
        st.write('선택한 컬럼이 없습니다.')




