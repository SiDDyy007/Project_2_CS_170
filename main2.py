import numpy as np

def calculate_accuracy(data, current_set_features):
    number_correctly_classified = 0
    # max_acc = 0
    for i in range(data.shape[0]):
        object_to_classify = data[i, current_set_features]
        label_object_to_classify = data[i, 0]

        # print(f'Looping over i, at the {i + 1} location')
        # print(f'The {i + 1}th object is in class {label_object_to_classify}')
        # print
        nearest_neighbor_distance = np.inf
        nearest_neighbor_location = np.inf
        for k in range(data.shape[0]):
            if k != i:
                distance = np.sqrt(np.sum((object_to_classify - data[k, current_set_features]) ** 2))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
        nearest_neighbor_label = data[nearest_neighbor_location, 0]
        # print(f"We have predicted the class as {nearest_neighbor_label}\n")
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1
    accuracy = number_correctly_classified / data.shape[0]
    return accuracy
    # print(f"Accuracy obtained for leaving the {i}th object outside is -> {accuracy}")
    max_acc = max(max_acc, accuracy)
    return max_acc


# accuracy = calculate_accuracy()
# print('Accuracy:', accuracy)


data = np.loadtxt(r'small-test-dataset.txt')
current_set_features = [ 3, 5, 7]  # or whatever features you're interested in
accuracy = calculate_accuracy(data, current_set_features)
print(f'The accuracy of the classifier using the given subset of features is: {accuracy * 100}%')

