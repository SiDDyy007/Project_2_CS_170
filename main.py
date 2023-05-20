import numpy as np

def dummy_function(data, current_set_of_features, k):
    # This function returns a random value between 0 and 1.
    return np.random.random()

def feature_search_demo(num_features):
    # Define a dummy dataset with num_features and 10 instances
    data = np.random.rand(10, num_features + 1)
    
    current_set_of_features = []  # Initialize empty set
    # Welcome message
    print("Welcome to Siddhant's Feature Selection Algorithm.")
    print(f"Please enter total number of features: {data.shape[1] - 1}")
    print("Type the number of the algorithm you want to run.")
    print("1. Forward Selection\n2. Backward Elimination")
    print("1")

    # Using no features and “random” evaluation
    accuracy = dummy_function(data, current_set_of_features, 0)
    print(f"Using no features and “random” evaluation, I get an accuracy of {accuracy * 100}%")
    print("Beginning search.")

    # best_so_far_accuracy = 0
    for i in range(data.shape[1] - 1):
        print(f'\n On the {i + 1}th level of the search tree')
        feature_to_add_at_this_level = None
        best_so_far_accuracy = 0
        for k in range(data.shape[1] - 1):
            if k not in current_set_of_features:
                print(f'Considering adding the {k + 1} feature')
                current_set_of_features.append(k)
                accuracy = dummy_function(data, current_set_of_features, k)
                print(f'Using feature(s) {current_set_of_features} accuracy is {accuracy * 100}%')
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
                else:
                    current_set_of_features.remove(k)
        if feature_to_add_at_this_level is not None:
            print(f'Feature set {current_set_of_features} was best, accuracy is {best_so_far_accuracy * 100}%')

    print(f'\nFinished search!! The best feature subset is {current_set_of_features}, which has an accuracy of {best_so_far_accuracy * 100}%')

# Take number of features as input from user
num_features = int(input("Enter the number of features: "))

# Use the feature search demo function
feature_search_demo(num_features)
