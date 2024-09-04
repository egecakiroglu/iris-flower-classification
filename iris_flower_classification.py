import ctypes

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from KNN import Knn

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

print(y_pred)

accuracy = accuracy_score(Y_test, y_pred)

# print(data["species"].unique())

print(f"Accuracy: {accuracy}")

# print(classification_report(Y_test, y_pred, target_names=data["species"].unique()))


print("------------------------------------------------------")
# BURASI KNN.PY İÇİN

# veriyi np.array formatına dönüştürdüm
X_test_cut = np.array(X_test.values)
X_train_cut = np.array(X_train.values)
Y_train_cut = np.array(Y_train.values)


# gerekli objeyi oluşturup metodları çağırdım
knnn = Knn(number_of_neighbors=3, X_train=X_train_cut, Y_train=Y_train_cut)

answer = knnn.predict(X_test=X_test_cut)
# print(answer)

# answer bir numpy arrayi, onun içindeki lablellardan en çok geçeni seçtim
ans2 = np.array([])
versi_num = 0
virgi_num = 0
setos_num = 0

for i in answer[1:]:
    for j in i:
        if j == "Iris-versicolor":
            versi_num += 1
        elif j == "Iris-virginica":
            virgi_num += 1
        elif j == "Iris-setosa":
            setos_num += 1
        else:
            continue
    if versi_num > setos_num and versi_num > virgi_num:
        ans2 = np.append(ans2, "Iris-versicolor")
    elif virgi_num > setos_num and virgi_num > versi_num:
        ans2 = np.append(ans2, "Iris-virginica")
    else:
        ans2 = np.append(ans2, "Iris-setosa")
    versi_num = 0
    virgi_num = 0
    setos_num = 0

print(ans2)

# sonucun isabetini hesapladım
accuracy_2 = accuracy_score(Y_test, ans2)

print(f"Accuracy: {accuracy_2}")


print("------------------------------------------------------")
# BURASI KNN.CPP İÇİN
