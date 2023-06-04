import numpy as np
import time
import heapq

def euclidean_distance(a, b):
    """
    Compute Euclidean distance between two vectors a and b
    """
    return sum((e1-e2)**2 for e1, e2 in zip(a,b))**0.5

def normalize_data(data):
    """
    Normalize data to have zero mean and unit variance
    """
    mean = np.mean(data[:, 1:], axis=0)
    std = np.std(data[:, 1:], axis=0)
    std[std == 0] = 1  # To prevent division by zero
    data[:, 1:] = (data[:, 1:] - mean) / std
    return data

class NN_Classifier:
    """
    A k-nearest neighbors classifier with a flexible number of neighbors
    """
    def __init__(self, data, feature_subset=None, k=1):
        """
        Initialize classifier. If no subset of features is specified, use all features
        """
        self.data = normalize_data(data)
        if feature_subset is None:
            self.feature_subset = list(range(1, self.data.shape[1]))  
        else:
            self.feature_subset = feature_subset
        self.k = k

    def train(self):
        """
        Train the classifier using leave-one-out validation and compute accuracy
        """
        start_time = time.time()  # start timer
        number_correctly_classified = 0
        for i in range(self.data.shape[0]):
            object_to_classify = self.data[i, self.feature_subset]
            label_object_to_classify = self.data[i, 0]
            nearest_neighbors = [(np.inf, -1) for _ in range(self.k)]  # Initialize nearest_neighbors
            
            # For each instance in the data (excluding the one we're classifying), compute distance to object_to_classify
            for j in range(self.data.shape[0]):
                if j != i:
                    distance = euclidean_distance(object_to_classify, self.data[j, self.feature_subset])
                    if distance < nearest_neighbors[-1][0]:  # If distance is smaller than the current furthest neighbor
                        nearest_neighbors[-1] = (distance, j)
                        nearest_neighbors.sort(key=lambda x: x[0])  # Sort nearest_neighbors based on distance
            
            # Classify based on majority label of nearest neighbors
            labels = [self.data[j, 0] for _, j in nearest_neighbors]
            majority_label = max(set(labels), key=labels.count)
            if label_object_to_classify == majority_label:  # If classification is correct, increment number_correctly_classified
                number_correctly_classified += 1
                
        accuracy = number_correctly_classified / self.data.shape[0]  # Compute accuracy
        end_time = time.time()  # end timer
        elapsed_time = round(end_time - start_time, 4)
        print(f'Time taken to train the classifier: {elapsed_time*10} milliseconds')
        return accuracy

    def test(self, instance):
        """
        Classify a single instance using the k-nearest neighbors algorithm
        """
        start_time = time.time()  # start timer
        object_to_classify = self.data[instance, self.feature_subset]
        nearest_neighbors = [(-np.inf, -1)] * self.k  # Initialize a max-heap with negative infinities
        
        for j in range(self.data.shape[0]): 
                    # object_to_classify
            if j != instance:
                distance = euclidean_distance(object_to_classify, self.data[j, self.feature_subset])
                if -distance > nearest_neighbors[0][0]:  # If distance is smaller than the current furthest neighbor
                    # Pop the max (smallest in magnitude) and push the new distance
                    heapq.heapreplace(nearest_neighbors, (-distance, j))

        end_time = time.time()  # end timer
        elapsed_time = round(end_time - start_time, 4)
        print(f'Time taken to test the classifier: {elapsed_time*10} milliseconds')

        # Negate distances back to positive when extracting the labels
        labels = [self.data[-d, 0] for d, j in nearest_neighbors]
        majority_label = max(set(labels), key=labels.count)  # Classify based on majority label of nearest neighbors

        return majority_label


class Validator:
    """
    A class to validate the performance of the nearest neighbors classifier
    """
    def __init__(self, data, feature_subset=None):
        """
        Initialize validator. If no subset of features is specified, use all features
        """
        self.data = data
        if feature_subset is None:
            self.feature_subset = list(range(1, self.data.shape[1]))  
        else:
            self.feature_subset = feature_subset
        self.nn_classifier = NN_Classifier(self.data, self.feature_subset)  # Initialize the nearest neighbors classifier
    
    def validate(self):
        """
        Train the classifier and return the accuracy
        """
        accuracy = self.nn_classifier.train()  # Train the classifier and get the accuracy
        return accuracy  # Return the accuracy of the classifier


def get_dataset_input():
    """
    Function to get user input for dataset selection
    """
    while True:
        print ("\nWhich dataset would you like to use?\n1. Small dataset\n2. Large dataset\nEnter choice (1 or 2), or 'q' to quit: ")
        dataset_choice = input()
        if dataset_choice == '1':
            data = np.loadtxt(r'small-test-dataset.txt')
            return data
        elif dataset_choice == '2':
            data = np.loadtxt(r'large-test-dataset-1.txt')
            return data
        elif dataset_choice.lower() == 'q':
            return None
        else:
            print("Invalid choice. Please enter 1 for small dataset or 2 for large dataset.")

def feature_input():
    """
    Function to get user input for feature selection
    """
    current_set_features = input("\nPlease enter the features you would like to train on, separated by commas: ")
    current_set_features = [int(x) for x in current_set_features.split(',')]
    return current_set_features
