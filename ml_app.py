# Libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.write("""# ML App
## Developed by: **Ikram Bashir**""")

# side bar for different dataset
dataset_name = st.sidebar.selectbox("Select Dataset", 
        ("Iris", "Wine", "Breast Cancer"))

# sidebar for different models
classifier_name = st.sidebar.selectbox("Select Classifier",
        ("KNN", "SVM", "Random Forest"))

# Function define for different dataset
def get_dataset(dataset_name):
    data = None
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x, y

# data structure
x, y = get_dataset(dataset_name)
st.write('Shape of the dataset:', x.shape)
st.write('number of classes:', len(np.unique(y)))

# function define for different classifier
def add_parameter_ui(classifier_name):
    params = dict()   # create an empty dictionary
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C    # degee of correct calssifier
    elif classifier_name == "KNN":
        k = st.sidebar.slider('k', 1, 15)
        params['k'] = k # number of neighbors
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth # depth of the tree in RF
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators # number of trees in RF
    return params

# call the function
params = add_parameter_ui(classifier_name)

# classifier function based on clasifier name
def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(**params)
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['k'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
            max_depth=params['max_depth'], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)

# spliting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# training of classifier
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# accuracy score of the classifier
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier= {classifier_name}')
st.write(f'Accuracy= {acc}')

### Plot the data ###
# 2 dimensional plot
pca = PCA(2)
x_projected = pca.fit_transform(x)

# 1 Dimension plot
x1 = x_projected[:, 0]
x2 = x_projected[:, 1]

fig = plt.figure(figsize=(6, 6))
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')

plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.colorbar()

#show plot
st.pyplot(fig)