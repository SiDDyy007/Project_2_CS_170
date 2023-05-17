import numpy as np

def dummy_function(data, current_set_of_features, k):
    # This function returns a random value between 0 and 1.
    return np.random.random()

def feature_search_demo(data):
    current_set_of_features = []  # Initialize empty set
    # best_so_far_accuracy = 0
    for i in range(data.shape[1] - 1):
        print(f'\n On the {i}th level of the search tree \n')
        feature_to_add_at_this_level = None
        best_so_far_accuracy = 0
        for k in range(data.shape[1] - 1):
            if k not in current_set_of_features:
                print(f'Considering adding the {k} feature')
                accuracy = dummy_function(data, current_set_of_features, k)
                print('Accuracy obtained -> ',accuracy*100, '%')
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
        if feature_to_add_at_this_level is not None:
            current_set_of_features.append(feature_to_add_at_this_level)
    return current_set_of_features

# Define a dummy dataset with 4 features and 10 instances
dummy_data = np.random.rand(10, 5)

# Use the feature search demo function on the dummy data
result = feature_search_demo(dummy_data)
print(f'Selected features: {result}')
