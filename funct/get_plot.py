import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from funct.get_pca import reduce_dimensions


def get_plot_data(x, y, loc, size=(5, 3), x_label='X', y_label='Y', plot_style='dark_background', legend_title='Class'):
    """
    Plots the reduced dimension data.
    Parameters:
    x (array-like): The high-dimensional data.
    y (array-like): The class labels.
    reducer (function): Function to reduce dimensions of x.
    loc (object): Object to display the plot.
    size (tuple): Size of the plot.
    x_label (str): Label for the x-axis.
    y_label (str): Label for the y-axis.
    plot_style (str): Style of the plot.
    legend_title (str): Title for the legend.

    Returns:
    None
    """
    x_red = reduce_dimensions(x)
    fig, ax = plt.subplots(figsize=size)
    plt.style.use(plot_style)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_labels = dict(zip(le.classes_, le.transform(le.classes_)))
    for label in np.unique(y):
        indices = np.where(y_encoded == class_labels[label])[0]
        ax.scatter(x_red[indices, 0], x_red[indices, 1], label=label)
    ax.legend(title=legend_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    loc.pyplot(fig)
