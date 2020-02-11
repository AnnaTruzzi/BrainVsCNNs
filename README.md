# Do the brain and CNNs similarly represent visual stimuli?
In this project we compared the internal represnetations of differnt trained and untrained network in response to 92 images with the neural activity pattern in the inferior temporal (IT) cortex of 15 subjects in response to the same stimuli. Specifically, we evaluated two versions of AlexNet and two training regimes, supervised and unsupervised, recording the activity at the output to the ReLU of the five convolutional layers and the two fully connected layers.

Stimuli and MRI data are available at the <a href="http://algonauts.csail.mit.edu/">algonauts project 2019</a>.

>Radoslaw Martin Cichy, Gemma Roig, Alex Andonian, Kshitij Dwivedi, Benjamin Lahner, AlexLascelles, Yalda Mohsenzadeh, Kandan Ramakrishnan, and Aude Oliva.  The algonauts project:A platform for communication between the sciences of biological and artificial intelligence.arXivpreprint arXiv:1905.05675, 2019.  

The code used to get the networks' activations in response to each image is available in this repository the following folders:
* deepcluster
* modified_alexnet
* standard_alexnet

For each netwokr's layer, we then characterised the representations through the representational dissimilarity matrix (RDM) and the correlation between each CNN's layer RDM and each human subject's RDM was calculated using the Mantel procedure with 10,000 permutations and the Kendall's Tau as statistic (comparison_with_the_brain.py).  

A repeated-measures ANOVA was then calculated with the Kendall's Tau values from every subject as dependent variable and the network type and layer as within-subject factors. As post-hocs, Student's t-tests were used to calculate whether the corresponding layers of different CNNs correlated with IT to a different extent, and whether within each CNN the representation in the last layer better correlated to IT compared with the first layer (2020ICLR_Analysis_Figures.py).

## Installation
**Requirements to run the <a href="https://github.com/facebookresearch/deepcluster">DeepCluster model</a> differ from the one necessary to run the rest of the repository.**
The code within the deepcluster folder was tested on python version 2.7.  
The rest of the repo was tested on python 3.7

## Get the data
Data are available in an open S3 bucket. To download them use the following command:  
`aws s3 cp s3://cusacklab/2020ICLR_BAICS_ATRC_opendata/ your/path/to/folder --recursive`  
  
What will be downloaded:
1. The trained unsupervised DeepCluster and supervised AlexNet
2. The activations of each network in response to the 92 images
3. The values (Kendall's Tau) of the correlations between each subject and each network's layer.
