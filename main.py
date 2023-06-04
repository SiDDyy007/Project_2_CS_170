import numpy as np
from part2 import NN_Classifier


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
    


def feature_search_demo(data):
    # Load the dataset from the file
    # data = np.loadtxt(data_file)
    
    num_instances, num_features = data.shape
    num_features -= 1  # Deduct one to exclude the class label column

    # Initialize feature set for both forward selection and backward elimination
    forward_features = []  
    backward_features = list(range(1, num_features + 1))  # 1-based index for features

    print(f"Your dataset has {num_features} features with {num_instances} instances.")
    print("Type the number of the algorithm you want to run.")
    print("1. Forward Selection\n2. Backward Elimination")
    
    choice = int(input())

    if choice == 1:
        print("Running Forward Selection")
        classifier = Validator(data, forward_features)
        features = forward_features
        print(f"Using no features (default rate) and training the classifier, I get an accuracy of {classifier.validate() * 100}%")
    elif choice == 2:
        print("Running Backward Elimination")
        classifier = Validator(data, backward_features)
        features = backward_features
        print(f"Using all features and training the classifer, I get an accuracy of {classifier.validate() * 100}%")
    else:
        print("Invalid choice. Please select either 1 or 2.")
        return
    # return
    print("Beginning search...")
    search_algorithm(data, features, choice, classifier)


def search_algorithm(data, features, choice, classifier):
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
                classifier = Validator(data, features)
                accuracy = classifier.validate()
                print(f'Using feature(s) {features} accuracy is {accuracy * 100}%')

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_change_at_this_level = k+1               
                features.remove(k+1) if choice == 1 else features.append(k+1)

        if feature_to_change_at_this_level is not None:
            res.append(feature_to_change_at_this_level) if choice == 1 else res.remove(feature_to_change_at_this_level)
            print(f'Feature set {res} was best, accuracy is {best_so_far_accuracy * 100}%')
        else:
            print("Accuracy has started to decrease... Hence ending the search ")
            accuracy_decreased = True
            break

    print(f'\nFinished search!! The best feature subset is {res}, which has an accuracy of {best_so_far_accuracy * 100}%')

def get_dataset_input():
    print ("\nWhich dataset would you like to use?\n1. Small dataset\n2. Large dataset\n3. Personal Small Dataset(20)\n4. Personal Large Dataset(20)")
    dataset_choice = input()
    if dataset_choice == '1':
        data = np.loadtxt(r'small-test-dataset.txt')
        return data
    elif dataset_choice == '2':
        data = np.loadtxt(r'large-test-dataset-1.txt')
        return data
    elif dataset_choice == '3':
        data = np.loadtxt(r'CS170_Spring_2023_Small_data__20.txt')
        return data
    elif dataset_choice == '4':
        data = np.loadtxt(r'CS170_Spring_2023_Large_data__20.txt')
        return data
    else:
        print("Invalid choice. Exiting the program....")
            

def start():
    print("Welcome to Siddhant's Feature Selection Algorithm:")
    data = get_dataset_input()
    choice = feature_search_demo(data)
            
start()