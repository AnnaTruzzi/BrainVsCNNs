import os
import scipy.io
import pickle
import numpy as np
from itertools import compress
from scipy.spatial import distance
from sklearn.manifold import MDS
from statsmodels.stats.anova import AnovaRM
import matplotlib
from matplotlib import pyplot as plt
import re
from scipy.cluster.hierarchy import dendrogram, linkage
from itertools import combinations
import scipy.io
import h5py
import hdf5storage
from scipy import stats
from scipy.spatial.distance import squareform
import collections
import re
import skbio
import seaborn as sns
import pingouin as pg
import pandas as pd
import glob
from sklearn.linear_model import LinearRegression
import scipy
from scipy.stats import pearsonr, spearmanr, kendalltau


## load activations dictionary 
def load_dict(path):
    with open(path,'rb') as file:
        dict=pickle.load(file, encoding="latin1")
    return dict

def reorder_od(dict1,order):
   new_od = collections.OrderedDict([(k,None) for k in order if k in dict1])
   new_od.update(dict1)
   return new_od
   
def loadmat(matfile):
    try:
        f = h5py.File(matfile)
    except (IOError, OSError):
        return io.loadmat(matfile)
    else:
        return {name: np.transpose(f.get(name)) for name in f.keys()}

def presentation_order(ordered_names_dict):
    orderedNames = []
    for item in sorted(ordered_names_dict.items()):
        orderedNames.append(item[1])
    return orderedNames

def rdm(act_matrix):
    rdm_matrix = distance.squareform(distance.pdist(act_matrix,metric='correlation'))
    return rdm_matrix


def mantel(x,y,out_file_name,training,corr_method,layer,brain_area):
    corr_value_list = []
    p_value_list = []
    for subj in range(0,y.shape[0]):
        y_subj = squareform(squareform(y[subj],checks = False))
        corr_value, p_value, n_value = skbio.stats.distance.mantel(x,y_subj, method = 'kendalltau', permutations = 10000)
        corr_value_list.append(corr_value)
        p_value_list.append(p_value)
    #####  save whole list of values for violin plots
    pickle_name = "singlesubj_corr_" + out_file_name + "_" + training + "_" + corr_method + 'only' + "_" + str(layer) + "_" + brain_area + '.pickle'
    with open('/your/path/to/folder/' + pickle_name, 'wb') as handle:
        pickle.dump(corr_value_list, handle)
    corr = np.mean(np.array(corr_value_list))
    p = np.mean(np.array(p_value_list))
    return corr,p

def rdm_plot(rdm, vmin, vmax, labels, main, outname):
   fig, ax = plt.subplots(figsize=(20,20))
   sns.heatmap(rdm, ax=ax, cmap='Blues_r', vmin=vmin, vmax=vmax, xticklabels=labels, yticklabels=labels)
   plt.show()
   fig.savefig('/your/path/to/folder/' + outname)
   plt.close()



def main(layers, network_used, stimuli, order_method,training,corr_method):
    ''' Calcutates the correlation between activations in a neural network and brain activations in IT

    Here is is applied to activations from 3 networks, either trained or untrained on the seto of 92 images from the Algonauts project.
    However it is also possible to extend the application to other sets of stimuli or networks by adding them to the list.

    Args:
        layers: list of layers of the network for which we have activations
        network_used: here it loops through DC, modified alexnet, and standard torchvision alexnet architectures 
        stimuli: here the considered stimuli are the images from the 92 images set in the Algonauts project
        order_method: here is the ordere of presentations in the mri.
        training: choose between pretrained vs untrained
        corr_method: here only the Mantel procedure is applied however it is possible to add other statisticsl procedures
    
    Output:
        Saves RDMs and correlations between each layers of the neural network and each subject IT activations.
    '''
    
    ### Set paths and load activations from the network and a dictionary with images names (to use as labels in the rdm plots)
    act_pth = '/home/CUSACKLAB/annatruzzi/cichy2016/' + stimuli + '_activations_'+ training + '_' + network_used + '.pickle'
    img_names_pth = '/home/CUSACKLAB/annatruzzi/cichy2016/' + stimuli + '_img_names.pickle'
    fmri_pth = '/home/CUSACKLAB/annatruzzi/cichy2016/algonautsChallenge2019/Training_Data/92_Image_Set/target_fmri.mat'

    act = load_dict(act_pth)
    img_names = load_dict(img_names_pth)
    img_names = reorder_od(img_names, sorted(img_names.keys()))


    ### get list of ordered labels starting from img_names dict
    order_function = order_method + '_order(img_names)'
    orderedNames = eval(order_function)
    orderedKeys = []
    for name in orderedNames:
        number = [i[0] for i in img_names.items() if i[1] == name]
        key = [k for k in act.keys() if '%s.jpg' %number[0] in k]
        orderedKeys.append(key[0])

    ### order activations dictionary according to the provided list of labels and 
    ### reorganize it in order to get one dictionary per layer and then image (instead of one dictionary per image and then layer)
    ordered_act = reorder_od(act,orderedKeys)
    
    layer_dict = collections.OrderedDict()
    for layer in range(0,len(layers)):
        layer_dict[layer] = collections.OrderedDict()
        print(layer)
        for item in ordered_act.items():
            if layer < 5:
                layer_dict[layer][item[0]] = np.mean(np.mean(np.squeeze(item[1][layer]),axis = 2), axis = 1)
            else:
                layer_dict[layer][item[0]] = np.squeeze(item[1][layer])


    ####### transform layer activations in matrix and calculate rdms
    layer_matrix_list = []
    net_rdm = []
    for l in layer_dict.keys():
        curr_layer = np.array([layer_dict[l][i] for i in orderedKeys])
        layer_matrix_list.append(curr_layer)
        net_rdm.append(rdm(curr_layer))


    ###### load fmri rdms 
    fmri_mat = loadmat(fmri_pth)
    IT = fmri_mat['IT_RDMs']

    ######### evaluate network vs fmri for each layer and subject
    out_file_name = '_'.join([network_used, stimuli])
    for layer in range(0,len(layers)):
        IT_corr,IT_pvalue = mantel(net_rdm[layer], IT, out_file_name, training, corr_method, layer, 'IT')


    ####### Create plots and save RDMs for the network
    for i,layer in enumerate(layers):
        main = network_used + ' layer '+str(layer)
        outname = '_'.join(['rdm',out_file_name, order_method,layer, training])
        print(outname)
        rdm_plot(net_rdm[i], vmin = 0, vmax = 1, labels = orderedNames, main = main, outname = outname + '.png')
        with open(('/your/path/to/folder/'+ outname + '_values.pickle'), 'wb') as handle:
            pickle.dump(net_rdm[i], handle)
    
    ####### Create plots and save RDM for the averaged mri data
    rdm_plot(np.mean(IT,axis = 0), vmin = 0, vmax = 0.8, labels = orderedNames, main = 'IT', outname = '_'.join(['rdm_IT',out_file_name,order_method]) + '.png')
    with open(('/your/path/to/folder/'+ '_'.join(['rdm_IT',stimuli ,order_method]) + '_values.pickle'), 'wb') as handle:
        pickle.dump(np.mean(IT,axis = 0), handle)



if __name__ == '__main__':
    
    layers_DC = ['ReLu1', 'ReLu2', 'ReLu3', 'ReLu4', 'ReLu5','ReLu6','ReLu7']
    layers_alexnet = ['ReLu1', 'ReLu2', 'ReLu3', 'ReLu4', 'ReLu5','ReLu6','ReLu7']
    net = ['DC','alexnet','torchalexnet']
    training = ['untrained','pretrained']

    for n in net:
        for t in training:
            print(n,t)
            main(eval('layers_' + n), n , 'niko92', 'presentation', t,'mantel')