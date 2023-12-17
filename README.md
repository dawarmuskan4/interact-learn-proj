<h1 align="center">🖥️ Interactive Accuracy Measurement Tool </h1>

<p align="center">
    This project is created to make it easier for us to perform Accuracy Measurement on different datasets. This project was created using Streamlit. 
</p>

## File Structure
```
.
├── images
├── functions
│   ├── custom_dataset.py
│   ├── get_best.py
│   ├── get_classifier.py
│   ├── get_data.py
│   ├── get_pca.py
│   ├── get_plot.py
│   ├── get_result.py
│   └── get_sidebar.csv
├── sample
│   ├── 2D.csv
│   └── 3D.csv
├── apps.py
├── Procfile
├── requirements.txt
├── setup.sh
.
```

## File Description

There are some important files in this repository, such as:

- `functions` folder contains various functions and procedures to maximize the performance and features of the
  dashboard.
- `sample` folder contains some sample data that can be used to test the file upload feature.
- `apps.py` is the python file to run our web app in Streamlit.

## How to Use
To run the project locally, you can download this repo and type
Python 3.11 is recommended to run this project.
```
pip install -r requirements.txt
streamlit run app.py
```

## References and Source
This project was inspired by:

- [Streamlit cheat sheet](https://streamlit-cheat-sheet.herokuapp.com/)
- [Membuat WebApp Machine Learning Interaktif by Afif Akbar](https://www.youtube.com/watch?v=_tbkwDGKfKQ&t=1905s)
- [Machine Learning Model Dashboard by Himanshu Sharma](https://www.youtube.com/watch?v=i0yrthZyiB8)
- [Membuat WebApp Machine Learning Interaktif](https://towardsdatascience.com/build-multiple-machine-learning-models-easily-54046f022483)
