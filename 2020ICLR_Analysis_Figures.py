from scipy import fftpack
import numpy as np
import pylab as py
import scipy.io
from PIL import Image
import os
import pandas as pd
import glob
import re
import pickle
from scipy.spatial import distance
import matplotlib
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform
from itertools import combinations
import pandas as pd
import seaborn as sns
import collections
from statsmodels.stats.anova import AnovaRM

## load activations dictionary 
def load_dict(path):
    with open(path,'rb') as file:
        dict=pickle.load(file, encoding="latin1")
    return dict

def main(corr_pth):
    ''' Calculates ANOVAs and post-hoc tests by loading the pickle files
    containing the values of the correlation between each network layers and each subject's IT activations.    
    In the end it creates the final plots.

    Args:
        corr_pth: the path to the folder containing the pickle files with the correlation values.
    
    Output:
        Prints analysis results and shows and saves plots.
    '''

    net_list = []
    net_training_list = []
    stimuli_list = []
    training_list = []
    brain_area_list = []
    layers_list = []
    tau_list = []
    subj_list = []
    for filename in glob.glob(corr_pth+'*mantel*.pickle'):
        print(filename)
        tau = np.array(load_dict(filename))
        tau_list.extend(tau)

        net = filename.split('/')[-1].split('_')[2]
        stimuli = filename.split('/')[-1].split('_')[3]
        training = filename.split('/')[-1].split('_')[4]
        brain_area = filename.split('/')[-1].split('_')[-1].split('.')[0]
        layer = filename.split('/')[-1].split('_')[-2]
        net_list.extend(np.repeat(net,len(tau)))
        stimuli_list.extend(np.repeat(stimuli,len(tau)))
        training_list.extend(np.repeat(training,len(tau)))
        net_training_list.extend(np.repeat('_'.join([net,training]),len(tau)))
        brain_area_list.extend(np.repeat(brain_area,len(tau)))
        layers_list.extend(np.repeat(layer,len(tau)))
        subj_list.extend(['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14'])
    
    data = {'tau': np.array(tau_list,dtype=float), 'network': net_list, 'stimuli': stimuli_list, 
           'training': training_list, 'brain_area': brain_area_list, 'layer': layers_list, 'net_training': net_training_list, 'subj': np.array(subj_list)}
    df = pd.DataFrame(data)


    #########  ANOVAs

    ####  Figure 1 
    dfaovfig1 = df.loc[(df['stimuli'] == 'niko92') & (df['brain_area'] == 'IT') & ((df['net_training'] == 'DC_untrained') | (df['net_training'] == 'torchalexnetINonly_pretrained'))]
    aovfig1 = AnovaRM(dfaovfig1, 'tau', 'subj', within=['net_training', 'layer'])
    resfig1 = aovfig1.fit()
    print(resfig1)

    ####  Figure 21 
    dfaovfig21 = df.loc[(df['stimuli'] == 'niko92') & (df['brain_area'] == 'IT') & ((df['net_training'] == 'torchalexnet_untrained') | (df['net_training'] == 'torchalexnet_pretrained'))]
    aovfig21 = AnovaRM(dfaovfig21, 'tau', 'subj', within=['net_training', 'layer'])
    resfig21 = aovfig21.fit()
    print(resfig21)

    ####  Figure 22 
    dfaovfig22 = df.loc[(df['stimuli'] == 'niko92') & (df['brain_area'] == 'IT') & ((df['net_training'] == 'DC_untrained') | (df['net_training'] == 'DC_pretrained'))]
    aovfig22 = AnovaRM(dfaovfig22, 'tau', 'subj', within=['net_training', 'layer'])
    resfig22 = aovfig22.fit()
    print(resfig22)

    ####  Figure 22 
    dfaovfig23 = df.loc[(df['stimuli'] == 'niko92') & (df['brain_area'] == 'IT') & ((df['net_training'] == 'alexnet_untrained') | (df['net_training'] == 'alexnet_pretrained'))]
    aovfig23 = AnovaRM(dfaovfig23, 'tau', 'subj', within=['net_training', 'layer'])
    resfig23 = aovfig23.fit()
    print(resfig23)


    ### Prepare lists of mean values and standard error per network per layer in order to easily use them in the plots
    corr_mean_dc_untrained = []
    corr_mean_dc_trained = []
    corr_mean_dcalexnet_untrained = []
    corr_mean_dcalexnet_trained = []
    corr_mean_standardalexnet_untrained = []
    corr_mean_standardalexnet_trained = []
    corr_sem_dc_untrained = []
    corr_sem_dc_trained = []
    corr_sem_dcalexnet_untrained = []
    corr_sem_dcalexnet_trained = []
    corr_sem_standardalexnet_untrained = []
    corr_sem_standardalexnet_trained = []
    for l in range(0,7):
        corr_mean_dc_untrained.append(np.mean(np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'DC_untrained') & (df['layer'] == str(l)) & (df['brain_area'] == 'IT')]['tau'])))
        corr_mean_dc_trained.append(np.mean(np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'DC_pretrained') & (df['layer'] == str(l)) & (df['brain_area'] == 'IT')]['tau'])))
        corr_mean_dcalexnet_untrained.append(np.mean(np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'alexnet_untrained') & (df['layer'] == str(l)) & (df['brain_area'] == 'IT')]['tau'])))
        corr_mean_dcalexnet_trained.append(np.mean(np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'alexnet_pretrained') & (df['layer'] == str(l)) & (df['brain_area'] == 'IT')]['tau'])))
        corr_mean_standardalexnet_trained.append(np.mean(np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'torchalexnet_pretrained') & (df['layer'] == str(l)) & (df['brain_area'] == 'IT')]['tau'])))
        corr_mean_standardalexnet_untrained.append(np.mean(np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'torchalexnet_untrained') & (df['layer'] == str(l)) & (df['brain_area'] == 'IT')]['tau'])))

        corr_sem_dc_untrained.append(scipy.stats.sem(np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'DC_untrained') & (df['layer'] == str(l)) & (df['brain_area'] == 'IT')]['tau'])))
        corr_sem_dc_trained.append(scipy.stats.sem(np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'DC_pretrained') & (df['layer'] == str(l)) & (df['brain_area'] == 'IT')]['tau'])))
        corr_sem_dcalexnet_untrained.append(scipy.stats.sem(np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'alexnet_untrained') & (df['layer'] == str(l)) & (df['brain_area'] == 'IT')]['tau'])))
        corr_sem_dcalexnet_trained.append(scipy.stats.sem(np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'alexnet_pretrained') & (df['layer'] == str(l)) & (df['brain_area'] == 'IT')]['tau'])))
        corr_sem_standardalexnet_trained.append(scipy.stats.sem(np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'torchalexnet_pretrained') & (df['layer'] == str(l)) & (df['brain_area'] == 'IT')]['tau'])))
        corr_sem_standardalexnet_untrained.append(scipy.stats.sem(np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'torchalexnet_untrained') & (df['layer'] == str(l)) & (df['brain_area'] == 'IT')]['tau'])))


    layers = ['0','1','2','3','4','5','6']
    layers_combinations = list(combinations(layers,2))

    #######################
    ########  FIG1  #######
    #######################
    p_fig1_between = []
    layers = np.arange(0,7)
    for l in layers:
        tau_dc_untrained = np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'DC_untrained') & (df['layer'] == str(l)) & (df['brain_area'] == 'IT')]['tau'])
        tau_standardalexnet_trained = np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'torchalexnet_pretrained') & (df['layer'] == str(l)) & (df['brain_area'] == 'IT')]['tau'])
        stat, p_value = scipy.stats.ttest_ind(tau_dc_untrained,tau_standardalexnet_trained)
        p_fig1_between.append(p_value)


    fig, ax = plt.subplots(figsize=(10,10))
    x = ['L1','L2','L3','L4','L5','L6','L7']
    y1 = np.array(corr_mean_dc_untrained)
    sem1 = np.array(corr_sem_dc_untrained)
    y2 = np.array(corr_mean_standardalexnet_trained)
    sem2 = np.array(corr_sem_standardalexnet_trained)
    ax.set_ylim(-0.02,0.4)
    ax.tick_params(labelsize=15)
    ax.set_ylabel('Kendall Tau',fontsize=20)
    ax.set_xlabel('Network Layers',fontsize=20)
    ax.plot(x,y1,linestyle='dashed',label="DeepCluster_untrained",marker='o',color = sns.xkcd_rgb['denim blue'])
    ax.fill_between(x, y1-sem1, y1+sem1, alpha = 0.4,color = sns.xkcd_rgb['denim blue'])
    ax.plot(x,y2,label="Stardard AlexNet_trained",marker='o', color = sns.xkcd_rgb["leaf green"])
    ax.fill_between(x, y2-sem2, y2+sem2, alpha = 0.4,color = sns.xkcd_rgb['leaf green'])
    ax.axhline(0.38, color='grey', lw=2, alpha=0.4)
    ax.axhline(0.31, color='gray', lw=2, alpha=0.4)
    ax.axhspan(0.31, 0.38, facecolor='gray', alpha=0.4)
    p_fig1_between = np.array(p_fig1_between)
    for i in range(0,len(p_fig1_between)):
        if p_fig1_between[i] < 0.05:
            ax.text(i-0.05,(y1[i]+y2[i])/2 - 0.008,'*',fontsize=20)
    ax.legend(loc='center right', prop={'size': 15},bbox_to_anchor=(1,0.6))
    ax.plot([0,0, 6,6], [0.022,0.02,0.02, 0.022], lw=1.5, color = 'black')
    ax.text((0+6)*.5, 0.002, "*", ha='center', va='bottom',fontsize = 20)
    plt.show()
    fig.savefig('2020ICLR_Fig1.pdf',bbox_inches='tight')
    plt.close()



    #######################
    ########  FIG2  #######
    #######################

    ### 2.1
    p_fig21_between = []
    layers = np.arange(0,7)
    for l in layers:
        tau_standardalexnet_untrained = np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'torchalexnet_untrained') & (df['layer'] == str(l)) & (df['brain_area'] == 'IT')]['tau'])
        tau_standardalexnet_trained = np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'torchalexnet_pretrained') & (df['layer'] == str(l)) & (df['brain_area'] == 'IT')]['tau'])
        stat, p_value = scipy.stats.ttest_ind(tau_standardalexnet_untrained,tau_standardalexnet_trained)
        p_fig21_between.append(p_value)

    tau_trained_0 = np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'torchalexnet_pretrained') & (df['layer'] == '0') & (df['brain_area'] == 'IT')]['tau'])
    tau_trained_6= np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'torchalexnet_pretrained') & (df['layer'] == '6') & (df['brain_area'] == 'IT')]['tau'])
    stat, p_value = scipy.stats.ttest_ind(tau_trained_0,tau_trained_6)
#    print('torchalexnet_pretrained')
#    print(stat)
#    print(p_value)

    tau_untrained_0 = np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'torchalexnet_untrained') & (df['layer'] == '0') & (df['brain_area'] == 'IT')]['tau'])
    tau_untrained_6= np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'torchalexnet_untrained') & (df['layer'] == '6') & (df['brain_area'] == 'IT')]['tau'])
    stat, p_value = scipy.stats.ttest_ind(tau_untrained_0,tau_untrained_6)
#    print('torchalexnet_untrained')
#    print(stat)
#    print(p_value)
  

    ### 2.2
    p_fig22_between = []
    layers = np.arange(0,7)
    for l in layers:
        tau_dc_untrained = np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'DC_untrained') & (df['layer'] == str(l)) & (df['brain_area'] == 'IT')]['tau'])
        tau_dc_trained = np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'DC_pretrained') & (df['layer'] == str(l)) & (df['brain_area'] == 'IT')]['tau'])
        stat, p_value = scipy.stats.ttest_ind(tau_dc_untrained,tau_dc_trained)
        p_fig22_between.append(p_value)

    tau_trained_0 = np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'DC_pretrained') & (df['layer'] == '0') & (df['brain_area'] == 'IT')]['tau'])
    tau_trained_6= np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'DC_pretrained') & (df['layer'] == '6') & (df['brain_area'] == 'IT')]['tau'])
    stat, p_value = scipy.stats.ttest_ind(tau_trained_0,tau_trained_6)
#    print('DC_pretrained')
#    print(stat)
#    print(p_value)

    tau_untrained_0 = np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'DC_untrained') & (df['layer'] == '0') & (df['brain_area'] == 'IT')]['tau'])
    tau_untrained_6= np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'DC_untrained') & (df['layer'] == '6') & (df['brain_area'] == 'IT')]['tau'])
    stat, p_value = scipy.stats.ttest_ind(tau_untrained_0,tau_untrained_6)
#    print('DC_untrained')
#    print(stat)
#    print(p_value)


    ### 2.3
    p_fig23_between = []
    layers = np.arange(0,7)
    for l in layers:
        tau_dcalexnet_untrained = np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'alexnet_untrained') & (df['layer'] == str(l)) & (df['brain_area'] == 'IT')]['tau'])
        tau_dcalexnet_trained = np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'alexnet_pretrained') & (df['layer'] == str(l)) & (df['brain_area'] == 'IT')]['tau'])
        stat, p_value = scipy.stats.ttest_ind(tau_dcalexnet_untrained,tau_dcalexnet_trained)
        print(l)
        print(stat)
        print(p_value)
        p_fig23_between.append(p_value)

    tau_trained_0 = np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'alexnet_pretrained') & (df['layer'] == '0') & (df['brain_area'] == 'IT')]['tau'])
    tau_trained_6= np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'alexnet_pretrained') & (df['layer'] == '6') & (df['brain_area'] == 'IT')]['tau'])
    stat, p_value = scipy.stats.ttest_ind(tau_trained_0,tau_trained_6)
#    print('alexnet_pretrained')
#    print(stat)
#    print(p_value)

    tau_untrained_0 = np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'alexnet_untrained') & (df['layer'] == '0') & (df['brain_area'] == 'IT')]['tau'])
    tau_untrained_6= np.array(df.loc[(df['stimuli'] == 'niko92') & (df['net_training'] == 'alexnet_untrained') & (df['layer'] == '6') & (df['brain_area'] == 'IT')]['tau'])
    stat, p_value = scipy.stats.ttest_ind(tau_untrained_0,tau_untrained_6)
#    print('alexnet_untrained')
#    print(stat)
#    print(p_value)

    
    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(20,20))
    x = ['L1','L2','L3','L4','L5','L6','L7']
    y1 = np.array(corr_mean_standardalexnet_trained)
    sem1 = np.array(corr_sem_standardalexnet_trained)
    y2 = np.array(corr_mean_standardalexnet_untrained)
    sem2 = np.array(corr_sem_standardalexnet_untrained)
    ax[0].set_ylim(-0.02,0.4)
    ax[0].tick_params(labelsize=15)
    ax[0].set_ylabel('Kendall Tau',fontsize=20)
    ax[0].set_xlabel('Network Layers',fontsize=20)
    ax[0].plot(x,y1,label="Stardard AlexNet_trained",marker='o',color = sns.xkcd_rgb["leaf green"])
    ax[0].fill_between(x, y1-sem1, y1+sem1, alpha = 0.4,color = sns.xkcd_rgb['leaf green'])
    ax[0].plot(x,y2,linestyle='dashed',label="Standard AlexNet_untrained",marker='o',color = sns.xkcd_rgb["leaf green"])
    ax[0].fill_between(x, y2-sem2, y2+sem2, alpha = 0.4,color = sns.xkcd_rgb['leaf green'])
    ax[0].axhline(0.38, color='grey', lw=2, alpha=0.4)
    ax[0].axhline(0.31, color='gray', lw=2, alpha=0.4)
    ax[0].axhspan(0.31, 0.38, facecolor='gray', alpha=0.4)
    p_fig21_between = np.array(p_fig21_between)
    for i in range(0,len(p_fig21_between)):
        if p_fig21_between[i] < 0.05:
            ax[0].text(i-0.08,(y1[i]+y2[i])/2 - 0.005,'*',fontsize=20)        
    ax[0].legend(loc='center right', prop={'size': 15},bbox_to_anchor=(1,0.6))
    ax[0].plot([0,0, 6,6], [0.138,0.14,0.14, 0.138], lw=1.5, color = 'black')
    ax[0].text((0+6)*.5, 0.14, "*", ha='center', va='bottom',fontsize = 20)


    x = ['L1','L2','L3','L4','L5','L6','L7']
    y1 = np.array(corr_mean_dc_trained)
    sem1 = np.array(corr_sem_dc_trained)
    y2 = np.array(corr_mean_dc_untrained)
    sem2 = np.array(corr_sem_dc_untrained)
    ax[1].set_ylim(-0.02,0.4)
    ax[1].tick_params(labelsize=15)
    ax[1].set_xlabel('Network Layers',fontsize=20)
    ax[1].plot(x,y1,label="DeepCluster_trained",marker='o',color = sns.xkcd_rgb["denim blue"])
    ax[1].fill_between(x, y1-sem1, y1+sem1, alpha = 0.4,color = sns.xkcd_rgb['denim blue'])
    ax[1].plot(x,y2,linestyle='dashed',label="DeepCluster_untrained",marker='o',color = sns.xkcd_rgb["denim blue"])
    ax[1].fill_between(x, y2-sem2, y2+sem2, alpha = 0.4,color = sns.xkcd_rgb['denim blue'])
    ax[1].axhline(0.38, color='grey', lw=2, alpha=0.4)
    ax[1].axhline(0.31, color='gray', lw=2, alpha=0.4)
    ax[1].axhspan(0.31, 0.38, facecolor='gray', alpha=0.4)
    p_fig22_between = np.array(p_fig22_between)
    for i in range(0,len(p_fig22_between)):
        if p_fig22_between[i] < 0.05:
            ax[1].text(i-0.08,(y1[i]+y2[i])/2 - 0.005,'*',fontsize=20)        
    ax[1].plot([0,0, 6,6], [0.022,0.02,0.02, 0.022], lw=1.5, color = 'black')
    ax[1].text((0+6)*.5, 0.002, "*", ha='center', va='bottom',fontsize = 20)
    ax[1].legend(loc='center right', prop={'size': 15},bbox_to_anchor=(1,0.6))

    x = ['L1','L2','L3','L4','L5','L6','L7']
    y1 = np.array(corr_mean_dcalexnet_trained)
    sem1 = np.array(corr_sem_dcalexnet_trained)
    y2 = np.array(corr_mean_dcalexnet_untrained)
    sem2 = np.array(corr_sem_dcalexnet_untrained)
    ax[2].set_ylim(-0.02,0.4)
    ax[2].tick_params(labelsize=15)
    ax[2].set_xlabel('Network Layers',fontsize=20)
    ax[2].plot(x,y1,label="Modified AlexNet_trained",marker='o',color = sns.xkcd_rgb["burgundy"])
    ax[2].fill_between(x, y1-sem1, y1+sem1, alpha = 0.4,color = sns.xkcd_rgb['burgundy'])
    ax[2].plot(x,y2,linestyle='dashed',label="Modified AlexNet_untrained",marker='o',color = sns.xkcd_rgb["burgundy"])
    ax[2].fill_between(x, y2-sem2, y2+sem2, alpha = 0.4,color = sns.xkcd_rgb['burgundy'])
    ax[2].axhline(0.38, color='grey', lw=2, alpha=0.4)
    ax[2].axhline(0.31, color='gray', lw=2, alpha=0.4)
    ax[2].axhspan(0.31, 0.38, facecolor='gray', alpha=0.4)
    p_fig23_between = np.array(p_fig23_between)
    for i in range(0,len(p_fig23_between)):
        if p_fig23_between[i] < 0.05:
            ax[2].text(i-0.08,(y1[i]+y2[i])/2 - 0.005,'*',fontsize=20)        
    ax[2].legend(loc='center right', prop={'size': 15},bbox_to_anchor=(1,0.6))
    ax[2].plot([0,0, 6,6], [0.022,0.02,0.02, 0.022], lw=1.5, color = 'black')
    ax[2].text((0+6)*.5, 0.002, "*", ha='center', va='bottom',fontsize = 20)
    plt.show()
    fig.savefig('2020ICLR_Fig2.pdf',bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    
    corr_pth = '/path/to/folder/with/correlation/values/for/single/subjects/'
    main(corr_pth)
