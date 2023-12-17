import numpy as np
import streamlit as st


@st.cache_data
def generate_xor_dataset():
    """Generates XOR dataset."""
    n_per_class = 50
    x = np.vstack([
        np.random.rand(n_per_class, 2) / 2 + 0.5,  # Top right quadrant
        np.random.rand(n_per_class, 2) / 2,        # Bottom left quadrant
        np.random.rand(n_per_class, 2) / 2 + np.array([[0, 0.5]]),  # Top left quadrant
        np.random.rand(n_per_class, 2) / 2 + np.array([[0.5, 0]])   # Bottom right quadrant
    ])
    y = np.array([0] * 100 + [1] * 100)
    return x, y


@st.cache_data
def generate_donut_dataset():
    """Generates Donut dataset."""
    n = 200
    n_per_class = n // 2
    r_inner, r_outer = 5, 10

    # Inner circle
    r1 = np.random.randn(n_per_class) + r_inner
    theta = 2 * np.pi * np.random.rand(n_per_class)
    x_inner = np.column_stack((r1 * np.cos(theta), r1 * np.sin(theta)))

    # Outer circle
    r2 = np.random.randn(n_per_class) + r_outer
    x_outer = np.column_stack((r2 * np.cos(theta), r2 * np.sin(theta)))

    x = np.vstack([x_inner, x_outer])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return x, y
