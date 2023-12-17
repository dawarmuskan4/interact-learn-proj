import streamlit as st
from sklearn.decomposition import PCA


@st.cache_data
def reduce_dimensions(data):
    """
    Reduce dimensions of the dataset using PCA.

    Parameters:
    data (array-like): The input data for dimensionality reduction.

    Returns:
    array-like: The dataset with reduced dimensions.
    """
    # Check if PCA is necessary
    if data.shape[1] > 2:
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data)
    else:
        reduced_data = data
    return reduced_data
