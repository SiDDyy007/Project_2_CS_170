import numpy as np

def train_classifier(data, current_set_features):
    number_correctly_classified = 0
    for i in range(data.shape[0]):
        object_to_classify = data[i, current_set_features]
        label_object_to_classify = data[i, 0]

        nearest_neighbor_distance = np.inf
        nearest_neighbor_location = np.inf
        for k in range(data.shape[0]):
            if k != i:
                distance = np.sqrt(np.sum((object_to_classify - data[k, current_set_features]) ** 2))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
        nearest_neighbor_label = data[nearest_neighbor_location, 0]
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1
    accuracy = number_correctly_classified / data.shape[0]
    return accuracy

def test_classifier(data, instance, current_set_features):
    object_to_classify = data[instance, current_set_features]
    nearest_neighbor_distance = np.inf
    nearest_neighbor_location = np.inf
    for k in range(data.shape[0]):
        if k != instance:
            distance = np.sqrt(np.sum((object_to_classify - data[k, current_set_features]) ** 2))
            if distance < nearest_neighbor_distance:
                nearest_neighbor_distance = distance
                nearest_neighbor_location = k
    return data[nearest_neighbor_location, 0]

def normalize_data(data):
    """
    This function normalizes the data to have zero mean and unit variance.
    """
    # We only normalize the features (not the labels), so we use data[:, 1:] 
    mean = np.mean(data[:, 1:], axis=0)
    std = np.std(data[:, 1:], axis=0)

    # Avoid division by zero
    std[std == 0] = 1

    # Normalize features
    data[:, 1:] = (data[:, 1:] - mean) / std

    return data

def get_input():
    print ("Which dataset would you like to use?\n1. Small dataset\n2. Large dataset\nEnter choice (1 or 2): ")
    dataset_choice = input()
    if dataset_choice == '1':
        data = np.loadtxt(r'small-test-dataset.txt')
        
    elif dataset_choice == '2':
        data = np.loadtxt(r'large-test-dataset-1.txt')
    else:
        print("Invalid choice. Please enter 1 for small dataset or 2 for large dataset.")
        exit(1)
    # Normalizing before training or testing
    data = normalize_data(data)
    feature_input(data)
    
def feature_input(data):
    current_set_features = input("Please enter the indices of the features you would like to train on, separated by commas: ")
    current_set_features = [int(x) for x in current_set_features.split(',')]
    accuracy = train_classifier(data, current_set_features)
    print(f'\nThe accuracy of the classifier using the given subset of features is: {accuracy * 100}%')
    classify = input("Do you want to test the classifier using an instance point?\n1. Yes\n2. No: ")
    if classify == '1':
        instance = int(input("Please specify the instance index you would like to classify: "))
        predicted_label = test_classifier(data, instance, current_set_features)
        print(f'\nThe predicted label for the {instance+1}th object is: {predicted_label}')
    else:
        print("Thank you! Program Done!")

get_input()
