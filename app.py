from funct.get_sidebar import get_sidebar, get_sidebar_xy
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder


def select_features_labels(df, expander):
    x_col = expander.multiselect('Feature(s)', df.columns.tolist())
    y_col = expander.multiselect('Label(s)', df.columns.tolist())
    if len(x_col) == 0 or len(y_col) == 0:
        expander.warning('Choose the Feature and Label')
        return None, None
    x_data = np.array(df[x_col])
    label = LabelEncoder()
    y_data = label.fit_transform(df[y_col])
    y_data = np.array(y_data)
    return x_data, y_data


def main():
    st.set_page_config(page_title='Interactive ML Dashboard', page_icon='images/ml.ico')

    # Sidebar
    st.sidebar.subheader('Dataset')
    data_upload = st.sidebar.file_uploader("Upload a Clean Dataset", type="csv")
    dataset_name = st.sidebar.selectbox('Or Choose Predefined Dataset',
                                        ('None', 'Iris', 'Cancer', 'Wine', 'Digits', 'XoR', 'Donut'), index=0)

    if data_upload and dataset_name != 'None':
        st.sidebar.warning('Please Choose Only One Dataset')
    elif data_upload:
        df = pd.read_csv(data_upload)
        df = df.dropna()
        if df.shape[1] < 3:
            st.sidebar.warning('Please Use Other Data with at Least 3 Columns (2 Features and 1 Label)')
        else:
            upload_expander = st.sidebar.expander('Choose the Feature and Label')
            x_data, y_data = select_features_labels(df, upload_expander)
            if x_data is not None and y_data is not None:
                dataset_name = data_upload.name[:-4].title()
                get_sidebar_xy(dataset_name, x_data, y_data)
    elif dataset_name != 'None':
        get_sidebar(dataset_name)
    else:
        # Home
        st.markdown("""
            <h1 style='text-align: center;'>Interact Learn</h1>
            <div style="font-size: 18px; text-align: center;">
                <p>
                    <strong>"Interact Learn"</strong> - a portal to machine learning for everyone. Effortlessly upload datasets, select features, and explore machine learning models. Our user-friendly interface makes it simple for all skill levels to dive into data analysis and discover insights.
                </p>
                <p>
                    Embrace the power of data science without the complexities. With <strong>"Interact Learn"</strong>, you're not just using a tool; you're embarking on a journey of learning and discovery.
                </p>
                <p>
                    Predict, analyze, and visualize with ease. Our platform supports various machine learning models, allowing you to see the impact of your choices in real time. Powered by <strong>Streamlit</strong>, experience the thrill of watching your predictions unfold.
                </p>
                <p>
                    <strong>Project by:</strong><br>
                    <img src="https://media.giphy.com/media/HQTYdpx1yhxWpugAi2/giphy.gif" width="50" height="50">
                    <a href="https://www.linkedin.com/in/dawarmuskan4/" target="_blank">Muskan Dawar</a>,
                    <a href="https://www.linkedin.com/in/sujalsethi44/" target="_blank">Sujal Sethi</a>,
                    <a href="https://www.linkedin.com/in/nimish-batra/" target="_blank">Nimish Batra</a>
                </p>
                <p>
                    <strong>Join us</strong> in making machine learning accessible and engaging. Discover the potential within your data today with <strong>"Interact Learn"</strong>.
                </p>
            </div>
        """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
