import streamlit as st
from funct.get_classifier import get_model
from funct.get_plot import get_plot_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Constants
TEST_SIZE = 0.25
RANDOM_STATE = 46
ACCURACY_THRESHOLD = 0.70


def train_classifier(classifier_name, x_train, y_train):
    """
    Trains the classifier with the given training data.
    """
    clf = get_model(classifier_name)
    clf.fit(x_train, y_train)
    return clf


def display_results(column, data, labels, title):
    """
    Displays the results in a Streamlit column with the given title.
    """
    column.markdown(f"<h4 style='text-align: center;'>{title}</h4>", unsafe_allow_html=True)
    column.markdown('####')
    get_plot_data(data, labels, column, (5, 4))


def display_accuracy(accuracy):
    """
    Displays the accuracy of the model.
    """
    st.markdown(f"<h2 style='text-align: center;'> {'Accuracy : ' + str(accuracy)}</h2>", unsafe_allow_html=True)
    if accuracy < ACCURACY_THRESHOLD:
        st.warning("Warning: Consider additional preprocessing and feature engineering.")


def get_result(classifier_name, x, y):
    """
    Trains the model with the given classifier and data, and displays the results.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    clf = train_classifier(classifier_name, x_train, y_train)
    y_predict = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)

    col1, col2 = st.columns(2)
    display_results(col1, x_test, y_test, "Actual")
    display_results(col2, x_test, y_predict, "Predicted")
    display_accuracy(accuracy)
