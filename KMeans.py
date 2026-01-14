from typing import Any, TypeAlias
import random as rand
import time as time
import numpy as np

class KMeans:
    # value_range contains a list of tuples where each tuple is a min and max value for each item in the vector
    def __init__(self, k:int, debug=False):
        self.k = k
        self.vector_length:int = 0
        self.classes:np.ndarray|None = None
        self.labels:np.ndarray|None = None
        rand.seed(time.time())
        self.debug=debug

    def init_from_data(self, data:np.ndarray, k_idx:int) -> None:
        kth_class = k_idx*self.vector_length
        random_idx = rand.randint(0, data.shape[0]-1)
        self.classes[kth_class:kth_class+self.vector_length] = data[random_idx]

    def SE(self, v1:np.ndarray, v2:np.ndarray) -> np.float32:
        diff = v1-v2
        np.multiply(diff, diff, out=diff)
        sum_squared_diff = np.sum(diff)
        return sum_squared_diff

    def MSE(self, v1:np.ndarray, v2:np.ndarray) -> np.float32:
        mse = np.dot(v1, -v2)
        return mse

    def closest_class(self, point:np.ndarray) -> tuple[float, int]:
        min_pair = (self.SE(point, self.classes[:self.vector_length]), 0)
        for k_idx in range(1, self.k):
            kth_class = k_idx*self.vector_length
            curr_diff = self.SE(point, self.classes[kth_class:kth_class+self.vector_length])
            #curr_diff = self.MSE(point, self.classes[kth_class:kth_class+self.vector_length])
            min_diff, _ = min_pair
            if (curr_diff < min_diff):
                min_pair = (curr_diff, k_idx)
        return min_pair

    def train(self, data:np.ndarray, labels:np.ndarray|None, epochs=10):
        valid_labels = labels is not None
        # setup np arrays
        # data is a (num_elements, vector length) array
        self.vector_length = data.shape[1]
        self.classes = np.zeros((self.k*self.vector_length), dtype=np.float32)
        for k_idx in range(self.k):
            self.init_from_data(data, k_idx)

        if (valid_labels):
            self.labels = np.zeros((self.k), dtype=np.int32)
            # {k_idx, {label, votes}}
            voting:dict[int, dict[int, int]] = {}

        for epoch in range(epochs):
            copy = np.zeros_like(self.classes)
            class_associations = np.zeros((self.k), dtype=np.int32)
            for point_idx in range(data.shape[0]):
                point = data[point_idx]
                _, closest_k_idx = self.closest_class(point)
                closest_class = closest_k_idx*self.vector_length
                copy[closest_class:closest_class+self.vector_length] += point
                class_associations[closest_k_idx] += 1

                if (epoch >= epochs-1 and valid_labels):
                    label_votes = voting.get(closest_k_idx)
                    if (label_votes is None):
                        label_votes = {}
                    label = labels[point_idx].item()
                    curr_votes = label_votes.get(label)
                    if (curr_votes is None):
                        label_votes[label] = 1
                    else:
                        label_votes[label] += 1
                    voting[closest_k_idx] = label_votes

            self.classes = copy
            for k_idx in range(self.k):
                kth_class = k_idx*self.vector_length
                num_associations = class_associations[k_idx]
                if (num_associations > 0):
                    self.classes[kth_class:kth_class+self.vector_length] /= num_associations
                else:
                    self.init_from_data(data, k_idx)

        if (valid_labels):
            for k_idx in range(self.k):
                label_votes = voting.get(k_idx)
                self.labels[k_idx] = -1
                if (label_votes is not None):
                    max_votes = (None, 0)
                    for label in label_votes.keys():
                        curr_votes = label_votes[label]
                        if (curr_votes > max_votes[1]):
                            max_votes = (label, curr_votes)
                    winning_label = max_votes[0]
                    self.labels[k_idx] = winning_label
                
    def predict(self, point:np.ndarray) -> int:
        distance, class_idx = self.closest_class(point)
        return class_idx
    
    def test(self, points:np.ndarray, labels:np.ndarray) -> float:
        if (self.labels is None):
            raise ValueError('Training data was not labeled.')
        acc:float = 0.0
        for point_idx in range(points.shape[0]):
            point = points[point_idx]
            k_idx = self.predict(point)
            predicted_label = self.labels[k_idx]
            actual_label = int(labels[point_idx])
            if (predicted_label == actual_label):
                acc += 1

        return (acc*100 / points.shape[0])