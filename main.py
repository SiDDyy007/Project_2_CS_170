import numpy as np

def dummy_function(data, current_set_of_features):
    # This function returns a random value between 0 and 1.
    return np.random.random()

def feature_search_demo(num_features):
    # Define a dummy dataset with num_features and 10 instances
    data = np.random.rand(10, num_features + 1)

    # Initialize feature set for both forward selection and backward elimination
    forward_features = []  
    backward_features = list(range(num_features)) 

    print(f"Welcome to Siddhant's Feature Selection Algorithm.")
    print(f"Total number of features: {num_features}")
    print("Type the number of the algorithm you want to run.")
    print("1. Forward Selection\n2. Backward Elimination")
    choice = int(input())

    if choice == 1:
        print("Running Forward Selection")
        features = forward_features
        print(f"Using no features and “random” evaluation, I get an accuracy of {dummy_function(data, []) * 100}%")
    else:
        print("Running Backward Elimination")
        features = backward_features
        print(f"Using all features and “random” evaluation, I get an accuracy of {dummy_function(data, backward_features) * 100}%")

    print("Beginning search.")
    search_algorithm(data, features, choice)

def search_algorithm(data, features, choice):
    accuracy_decreased = False
    res = features[:]
    best_so_far_accuracy = 0
    for i in range(data.shape[1] - 1):
        if accuracy_decreased:
            break
        print(f'\n On the {i + 1}th level of the search tree')
        feature_to_change_at_this_level = None
        
        features = res[:]
        for k in range(data.shape[1] - 1):
            if (choice == 1 and k+1 not in features) or (choice == 2 and k+1 in features):
                print(f'Considering {"adding" if choice == 1 else "removing"} the {k + 1} feature')
                features.append(k+1) if choice == 1 else features.remove(k+1)
                accuracy = dummy_function(data, features)
                print(f'Using feature(s) {features} accuracy is {accuracy * 100}%')

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_change_at_this_level = k+1
                # else:
                features.remove(k+1) if choice == 1 else features.append(k+1)

        if feature_to_change_at_this_level is not None:
            res.append(feature_to_change_at_this_level) if choice == 1 else res.remove(feature_to_change_at_this_level)
            print(f'Feature set {res} was best, accuracy is {best_so_far_accuracy * 100}%')
        else:
            print("Accuracy has started to decrease... Hence ending the search ")
            accuracy_decreased = True
            break

    print(f'\nFinished search!! The best feature subset is {res}, which has an accuracy of {best_so_far_accuracy * 100}%')

# Take number of features as input from user
num_features = int(input("Enter the number of features: "))

# Use the feature search demo function
feature_search_demo(num_features)
