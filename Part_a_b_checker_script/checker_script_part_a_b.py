#The objecive of this script is to check the weights for Part 1 (a) and (b) of the assignment.
#Ensure the weights of your Network are saved as mentioned in the coding guidelines of the assignment.
import numpy as np
import pickle

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def show_diff(yourfile, gtfile):
    your_wts = load_pickle(yourfile)
    gt_wts = load_pickle(gtfile)

    layer_keys = gt_wts["weights"].keys()

    for layer_key in layer_keys:
        your_wt1 = your_wts["weights"][layer_key]
        gt_wt1 = gt_wts["weights"][layer_key]

        b1 = your_wts["bias"][layer_key]
        b2 = gt_wts["bias"][layer_key]

        assert(your_wt1.shape == gt_wt1.shape)
        assert(b1.shape == b2.shape)

        print('layer            :' , layer_key)
        print('Submitted weight shape            :' , your_wt1.shape)
        print('Submitted bias shape               :',b1.shape)
        diff = np.abs(your_wt1 - gt_wt1) / (abs(gt_wt1)+0.00001)  #Tolerance term to avoid dividing by zero
        if np.sum(abs(b2)) == 0.0:
            diff_bias = np.abs(b1 - b2)
        else:
            diff_bias = np.abs(b1 - b2) / (abs(b2)+0.00001) #Tolerance term to avoid dividing by zero
        print('max relative wt diff:', 100 * np.max(diff), '%')
        print('max relative bias diff:', 100 * np.max(diff_bias), '%')

if __name__ == '__main__':
    your_weights_file = "weights.pkl"  #Add path to your weights.pkl here for a particular epoch.
    GT_weights_file = "Part_a_b_checker_script/Binary_xavier_updated_part_1_a/ep_5.pkl" #Add path to the given weights.pkl here for a particular epoch to be used for comparison.
    show_diff(your_weights_file, GT_weights_file)