import numpy as np
import pickle
import csv


def load_predictions(pickle_file):
    with open(pickle_file, 'rb') as f:
        your_predictions = pickle.load(f)  
    return np.array(your_predictions) 

def load_ground_truth(csv_file):
    ground_truth = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ground_truth.append(int(row['class'])) 
    return np.array(ground_truth)  

def cross_entropy_loss(predictions, ground_truth, num_classes):
    ground_truth_one_hot = np.eye(num_classes)[ground_truth]
    predictions_one_hot = np.eye(num_classes)[predictions]
    
    epsilon = 1e-12
    predictions_one_hot = np.clip(predictions_one_hot, epsilon, 1. - epsilon)
    
    cross_entropy = -np.sum(ground_truth_one_hot * np.log(predictions_one_hot)) / len(predictions)
    return cross_entropy

if __name__ == "__main__":
    # Example usage:
    
    #Save the predictions from your on "train set" as a predictions.pkl [Refer Assignment pdf for more details] file where it contains a numpy array of size equal to the number of training samples. 
    #Load the predictions array from the pickle file
    pickle_file = 'predictions.pkl' 
    your_predictions = load_predictions(pickle_file)
    
    #Load the ground truth labels from the CSV file
    csv_file = 'dataset_for_A2/multi_dataset/train.csv'  
    ground_truth = load_ground_truth(csv_file)
    
    #Assume the number of classes (num_classes) is known
    num_classes = len(np.unique(ground_truth))  
    
    #Compute and print the cross-entropy loss
    loss = cross_entropy_loss(your_predictions, ground_truth, num_classes)
    print(f'Cross-Entropy Loss: {loss}')
