import matplotlib.pyplot as plt
import numpy as np

# X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
# y_train = np.array([2, 3, 4, 5, 6])
# X_test = np.array([[2.5, 3.5], [4.5, 5.5]])


class Knn:
    def __init__(self, number_of_neighbors, X_train, Y_train):
        self.number_of_neighbors = number_of_neighbors
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, X_test):
        ans = np.array(["", "", ""])
        for x_test in X_test:
            dist = [np.linalg.norm(x_train - x_test) for x_train in self.X_train]
            # print(dist)

            k_indices = np.argsort(dist)[: self.number_of_neighbors]
            k_nearesst_neighbors = np.array([self.Y_train[i] for i in k_indices])
            # print(k_nearesst_neighbors)

            ans = np.vstack([ans, k_nearesst_neighbors])

        return ans


# knn = Knn(number_of_neighbors=3, X_train=X_train, Y_train=y_train)
# knn.predict(X_test=X_test)
