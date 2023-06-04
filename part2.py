import numpy as np
import time


def euclidean_distance(a, b):
    return sum((e1-e2)**2 for e1, e2 in zip(a,b))**0.5

def normalize_data(data):
    """
    This function normalizes the data to have zero mean and unit variance.
    """
    # return data
    mean = np.mean(data[:, 1:], axis=0)
    std = np.std(data[:, 1:], axis=0)
    std[std == 0] = 1
    data[:, 1:] = (data[:, 1:] - mean) / std
    return data

class NN_Classifier:
    def __init__(self, data, feature_subset=None):
        # Normalize data at initialization
        self.data = normalize_data(data)
        # If no subset of features is specified, use all features
        if feature_subset is None:
            self.feature_subset = list(range(1, self.data.shape[1]))  
        else:
            self.feature_subset = feature_subset    

    def train(self):
        # Start recording time for training
        start_time = time.time()
        number_correctly_classified = 0
        for i in range(self.data.shape[0]):
            # Select instance for classification
            object_to_classify = self.data[i, self.feature_subset]
            label_object_to_classify = self.data[i, 0]

            nearest_neighbor_distance = np.inf
            nearest_neighbor_location = np.inf
            # Loop through all data points
            for k in range(self.data.shape[0]):
                # Ensure the data point is not the one being classified
                if k != i:
                    # Compute the Euclidean distance between the data points
                    distance = euclidean_distance(object_to_classify, self.data[k, self.feature_subset])
                    # Update nearest neighbor if this data point is closer
                    if distance < nearest_neighbor_distance:
                        nearest_neighbor_distance = distance
                        nearest_neighbor_location = k
            # If the nearest neighbor has the same label, increment the count of correctly classified instances
            nearest_neighbor_label = self.data[nearest_neighbor_location, 0]
            if label_object_to_classify == nearest_neighbor_label:
                number_correctly_classified += 1
        # Compute the accuracy
        accuracy = number_correctly_classified / self.data.shape[0]

        # End recording time for training and print elapsed time
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 4)
        print(f'Time taken to train the classifier: {elapsed_time*10} milliseconds')
        
        return accuracy
    
    def test(self, instance):
        # Start recording time for testing
        start_time = time.time()
        object_to_classify = self.data[instance, self.feature_subset]
        nearest_neighbor_distance = np.inf
        nearest_neighbor_location = np.inf
        # Loop through all data points
        for k in range(self.data.shape[0]):
            # Ensure the data point is not the one being classified
            if k != instance:
                # Compute the Euclidean distance between the data points
                distance = euclidean_distance(object_to_classify, self.data[k, self.feature_subset])
                # Update nearest neighbor if this data point is closer
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
        # End recording time for testing and print elapsed time
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 4)
        print(f'Time taken to test the classifier: {elapsed_time*10} milliseconds')
        
        # Return the label of the nearest neighbor
        return self.data[nearest_neighbor_location, 0]


class Validator:
    def __init__(self, data, feature_subset=None):
        self.data = data
        # If no subset of features is specified, use all features
        if feature_subset is None:
            self.feature_subset = list(range(1, self.data.shape[1]))  
        else:
            self.feature_subset = feature_subset
        # Initialize the nearest neighbors classifier with the selected features
        self.nn_classifier = NN_Classifier(self.data, self.feature_subset)
    
    def validate(self):
        # Train the classifier and get the accuracy
        accuracy = self.nn_classifier.train()
        # Return the accuracy of the classifier
        return accuracy



# # def start():
#     while True:
#         choice = input("\nWhat would you like to do?\n1. Test the classifier using a given subset of features (Validator Class)\n2. Predict the class label for a given instance (Classifier Class)\nPlease enter choice (1 or 2), or 'q' to quit: ")
        
#         if choice == '1':
#             feature_subset = feature_input()
#             data = get_dataset_input()
#             if data is None:
#                 continue
#             validator = Validator(data, feature_subset)
#             accuracy = validator.validate()
#             print(f'\nThe accuracy of the classifier using the given subset of features is: {accuracy * 100}%')
#         elif choice == '2':
#             data = get_dataset_input()
#             if data is None:
#                 continue
#             instance = int(input("Please specify the instance ID (Starting from 1) you would like to classify: "))            
#             classifier = NN_Classifier(data)
#             predicted_label = classifier.test(instance-1)
#             print(f'\nThe predicted label for the {instance}th instance ID is: {predicted_label}')
#         elif choice.lower() == 'q':
#             print("Exiting the program. Goodbye!")
#             break
#         else:
#             print("Invalid choice. Please enter 1 to test the classifier or 2 to predict an instance.")

def get_dataset_input():
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
    current_set_features = input("\nPlease enter the features you would like to train on, separated by commas: ")
    current_set_features = [int(x) for x in current_set_features.split(',')]
    return current_set_features

# start()


