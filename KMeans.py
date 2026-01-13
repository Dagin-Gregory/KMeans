from typing import Any
import random as rand
import time as time
import numpy as np

class KMeans:
    # value_range contains a list of tuples where each tuple is a min and max value for each item in the vector
    def __init__(self, k:int, value_range:list[tuple[float,float]], debug=False):
        self.k = k
        self.vector_length = len(value_range)
        self.value_range = value_range
        self.classes:np.ndarray = np.zeros((k*self.vector_length), dtype=np.float32)
        self.labels:list[Any|None] = [None]*k
        rand.seed(time.time())
        self.debug=debug
    
    #def random_init(self, kth_class:int):
    #    for i in range(self.vector_length):
    #        min_val, max_val = self.value_range[i]
    #        percentage = rand.random() #0-1
    #        difference = max_val-min_val
    #        value = min_val + difference*percentage
    #        self.classes[kth_class][i] = value

    def MSE(self, v1:np.ndarray, v2:np.ndarray) -> float:
        if (v1.shape != v2.shape):
            raise ValueError('Lengths not the same')
        diff_arr = v1-v2
        np.multiply(diff_arr, diff_arr, out=diff_arr)
        mse = np.mean(diff_arr)
        return mse
    
    def L2Norm(self, v1:np.ndarray, v2:np.ndarray) -> float:
        if (v1.shape != v2.shape):
            raise ValueError('Lengths not the same')
        diff_arr = v1-v2
        np.multiply(diff_arr, diff_arr, out=diff_arr)
        se = np.sum(diff_arr)
        #l2 = np.sqrt(se)
        return se

    def closestClass(self, data:np.ndarray) -> tuple[float, int]:
        #min_mse_pair = (self.MSE(data, self.classes[0:self.vector_length]), 0)
        min_mse_pair = (self.L2Norm(data, self.classes[0:self.vector_length]), 0)
        for k_iter in range(1, self.k):
            kth_class = k_iter*self.vector_length
            #curr_mse = self.MSE(data, self.classes[kth_class:kth_class+self.vector_length])
            curr_mse = self.L2Norm(data, self.classes[kth_class:kth_class+self.vector_length])
            min_mse,_ = min_mse_pair
            if (curr_mse < min_mse):
                min_mse_pair = (curr_mse, k_iter)
        return min_mse_pair

    # We'll do regular K-means first just to get everything working then we can do batching
    #def train(self, data:list[list[Any]], labels:list[Any]|None=None, epochs=10, init_from_data=True, batch_size=1000):
    def train(self, data:np.ndarray, labels:np.ndarray|None=None, epochs=10, init_from_data=True, batch_size=1000):
        # Ignore batch_size for now

        if (init_from_data):
            taken_points:dict[int, bool] = {}
            for k_iter in range(self.k):
                kth_class = k_iter*self.vector_length
                random_index = int(rand.random()*len(data))
                while (taken_points.get(random_index, False)):
                    random_index = int(rand.random()*len(data))
                self.classes[kth_class:kth_class+self.vector_length] = data[random_index]
                taken_points[random_index] = True
        else:
            raise ValueError('Need labels passed in for now.')
            #for class_num in range(self.k):
            #    self.classes.append([0]*self.vector_length)
            #    self.random_init(class_num)

        # [(The data point, kth_class), ...]
        for epoch in range(1, epochs+1):
            time_start = time.time()
            point_associations:list[tuple[int,int]] = []
            total_associations:list[int] = [0]*self.k
            if (labels is not None):
                if (len(data) != len(labels)):
                    raise Exception('Length of data and length of labels is not equal.')
                # {kth_class, {Label, number of labels for that centroid}}
                voting:dict[int, dict[Any, int]] = {}
            else:
                labels = [0]*len(data)

            for idx in range(len(data)):
                element = data[idx]
                min_mse_pair = self.closestClass(element)
                _,k_iter = min_mse_pair
                kth_class = k_iter*self.vector_length
                point_associations.append((idx, kth_class))
                total_associations[k_iter] += 1

                label_dict = voting.get(k_iter, {})
                label_dict[labels[idx]] = label_dict.get(labels[idx], 0)+1
                voting[k_iter] = label_dict


            copy = np.zeros_like(self.classes)
            for point_idx,kth_class in point_associations:
                point = data[point_idx]
                #self.classes[kth_class:kth_class+self.vector_length] += point
                copy[kth_class:kth_class+self.vector_length] += point

            reinit = 0
            for k_iter in range(self.k):
                if (total_associations[k_iter] > 0):
                    kth_class = k_iter*self.vector_length
                    copy[kth_class:kth_class+self.vector_length] /= total_associations[k_iter]
                    #self.classes[kth_class:kth_class+self.vector_length] = copy[kth_class:kth_class+self.vector_length]
                else:
                #    self.random_init(curr_class)
                    reinit += 1
            self.classes = copy.copy()
            print(f"Reinitialized {reinit} points for epoch {epoch}")

            if (labels is not None):
                for curr_class in voting.keys():
                    max_pair:tuple[Any, int] = (0, 0)
                    curr_vote = voting[curr_class]
                    for label in curr_vote.keys():
                        _,max_val = max_pair
                        if (curr_vote[label] > max_val):
                            max_pair = (label, curr_vote[label])
                    label,_ = max_pair
                    self.labels[curr_class] = label

            if (self.debug):
                print(f"Time for iteration {epoch}: {time.time()-time_start:.2f}")
                time_start = time.time()

    def predict(self, data:np.ndarray) -> int:
        distance, kth_class = self.closestClass(data)
        return kth_class

    def test(self, data:np.ndarray, labels:np.ndarray) -> tuple[float, list[Any]]:
        if (len(data) != len(labels)):
            raise Exception('Length of data and labels are not equal')
        
        predictions:list[int] = []
        correct:float = 0.0
        for idx in range(len(data)):
            prediction = self.predict(data[idx])
            prediction_label = self.labels[prediction]
            predictions.append(prediction_label)
            if (prediction_label == labels[idx]):
                correct += 1
        
        return (correct/len(labels), predictions.copy())
    
    def printLabels(self):
        for label_num in range(len(self.labels)):
            print(f"Label for class {label_num}: {self.labels[label_num]}")