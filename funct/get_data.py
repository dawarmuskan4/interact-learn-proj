import numpy as np
import streamlit as st
from sklearn import datasets
from funct.custom_dataset import generate_xor_dataset, generate_donut_dataset
from typing import Tuple, Union

# Mapping of dataset names to their respective loader functions
DATASET_LOADERS = {
    'Iris': datasets.load_iris,
    'Cancer': datasets.load_breast_cancer,
    'Wine': datasets.load_wine,
    'Digits': datasets.load_digits,
    'XoR': generate_xor_dataset,
    'Donut': generate_donut_dataset
}


@st.cache_resource
def get_data(dataset_name: str) -> Union[Tuple[np.ndarray, np.ndarray], None]:
    """
    Load data based on the dataset name.

    Args:
    dataset_name (str): Name of the dataset to load.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Feature matrix (X) and target vector (y).
    If the dataset name is unknown, returns None.
    """
    if dataset_name in DATASET_LOADERS:
        if dataset_name in ['XoR', 'Donut']:
            x, y = DATASET_LOADERS[dataset_name]()
        else:
            dataset = DATASET_LOADERS[dataset_name]()
            x, y = dataset.data, dataset.target
        return x, y
    else:
        st.error(f"Dataset {dataset_name} is not recognized.")
        return None
