import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from funct.get_plot import get_plot_data


def get_best_classifier(x, y):
    # Define classifiers
    classifiers = {
        'k-NN': KNeighborsClassifier(),
        'SVC': SVC(),
        'Perceptron': Perceptron(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier()
    }

    # Train and test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=46)

    # Evaluate each classifier
    accuracies = {}
    for name, clf in classifiers.items():
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)
        accuracies[name] = accuracy_score(y_test, y_predict)

    # Determine the best classifier
    max_class = max(accuracies, key=accuracies.get)

    # Streamlit UI
    classifier_options = ['None'] + list(classifiers.keys())
    classifier_name = st.sidebar.selectbox('Choose Classifier', classifier_options,
                                           index=classifier_options.index(max_class))

    return classifier_name


def get_best_result(classifier_name, x, y):
    # Define classifiers
    classifiers = {
        'k-NN': KNeighborsClassifier(),
        'SVC': SVC(),
        'Perceptron': Perceptron(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier()
    }

    # Get classifier
    clf = classifiers.get(classifier_name)

    if clf is not None:
        # Train and test split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=46)

        # Train and predict
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)

        # Streamlit UI for results
        st.markdown("<h3 style='text-align: center;'>Comparison Between Actual and Predicted Labels</h3>",
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h4 style='text-align: center;'>Actual Labels</h4>", unsafe_allow_html=True)
            get_plot_data(x_test, y_test, col1, (5, 4))
        with col2:
            st.markdown("<h4 style='text-align: center;'>Predicted Labels</h4>", unsafe_allow_html=True)
            get_plot_data(x_test, y_pred, col2, (5, 4))

        st.markdown(f"<h2 style='text-align: center;'>Accuracy: {acc:.2f}</h2>", unsafe_allow_html=True)
