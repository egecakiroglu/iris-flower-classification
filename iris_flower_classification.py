import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("IRIS.csv")

# veriyi anlamak için açıklamalar
# print(data.head())
# print(data.describe())

# plotly kullanarak veriyi görselleştirme
# fig = px.scatter(data, x="sepal_width", y="sepal_length", color="species")
# fig.show()

# veriyi test ve train olarak ayırma

# label ı ana veriden ayırma
X = data.drop("species", axis=1)
Y = data["species"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# train verisini knn algoritmasına fit etme
# knn isimli object oluşturup onun fit fonksiyonunu kullandık
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)

# tahmin için ayırdığımız veri ile tahmin yapma
y_pred = knn.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)

# print(data["species"].unique())

print(f"Accuracy: {accuracy}")

print(classification_report(Y_test, y_pred, target_names=data["species"].unique()))
