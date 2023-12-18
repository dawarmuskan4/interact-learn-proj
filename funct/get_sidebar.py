import streamlit as st
from funct.get_best import get_best_classifier, get_best_result
from funct.get_data import get_data
from funct.get_result import get_result
from funct.get_plot import get_plot_data

CLASSIFIERS = ('None', 'k-NN', 'SVC', 'Perceptron', 'Decision Tree', 'Random Forest')


def display_results(classifier_name, dataset_name, x, y):
    st.markdown(
        f"<h1 style='text-align: center;'>{dataset_name + ' Dataset' + (' using ' + classifier_name if classifier_name != 'None' else '')}</h1>",
        unsafe_allow_html=True)
    if classifier_name != 'None':
        get_result(classifier_name, x, y)
    else:
        get_plot_data(x, y, st, (5, 3))


def handle_classifier_selection(dataset_name, x, y):
    best_button = st.sidebar.button('Find the Best Classifier Automatically')
    if best_button:
        classifier_name = get_best_classifier(x, y)
        get_best_result(classifier_name, x, y) if classifier_name != 'None' else display_results(classifier_name, dataset_name, x, y)
    else:
        classifier_name = st.sidebar.selectbox('Or Choose Classifier Manually', CLASSIFIERS, index=0)
    return classifier_name, best_button 


def get_sidebar(dataset_name):
    x, y = get_data(dataset_name)
    st.sidebar.subheader('Classifier')
    classifier_name, best_button = handle_classifier_selection(dataset_name, x, y)
    if classifier_name is not None and not best_button:
        display_results(classifier_name, dataset_name, x, y)


def get_sidebar_xy(dataset_name, x, y):
    st.sidebar.subheader('Classifier')
    classifier_name, best_button = handle_classifier_selection(dataset_name, x, y)
    if classifier_name is not None and not best_button:
        display_results(classifier_name, dataset_name, x, y)
