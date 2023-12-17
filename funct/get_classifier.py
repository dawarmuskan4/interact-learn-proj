import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_classifier_params(clf_name):
    """
    Returns the parameters for the specified classifier.
    """
    param_expander = st.sidebar.expander('Optional Parameter(s)')
    clf_params = {}
    if clf_name == 'k-NN':
        clf_params['n_neighbors'] = param_expander.slider('k', 1, 15)
    elif clf_name == 'SVC':
        clf_params['C'] = param_expander.slider('C', 0.01, 10.0)
        clf_params['kernel'] = param_expander.selectbox('Kernel', ('linear', 'poly', 'rbf', 'sigmoid'))
    elif clf_name == 'Perceptron':
        clf_params['penalty'] = param_expander.selectbox('Penalty', ('l1', 'l2', 'elasticnet', None))
    elif clf_name == 'Decision Tree':
        clf_params['criterion'] = param_expander.selectbox('criterion', ('gini', 'entropy'))
        clf_params['splitter'] = param_expander.selectbox('splitter', ('best', 'random'))
    elif clf_name == 'Random Forest':
        clf_params['n_estimators'] = param_expander.slider('n_estimators', 1, 100)
    return clf_params


def get_model(clf_name):
    """
    Creates and returns a classifier model based on the provided name.
    """
    params = get_classifier_params(clf_name)
    if clf_name == 'k-NN':
        return KNeighborsClassifier(**params)
    elif clf_name == 'SVC':
        return SVC(**params)
    elif clf_name == 'Perceptron':
        return Perceptron(**params)
    elif clf_name == 'Decision Tree':
        return DecisionTreeClassifier(**params)
    elif clf_name == 'Random Forest':
        return RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unknown classifier: {clf_name}")