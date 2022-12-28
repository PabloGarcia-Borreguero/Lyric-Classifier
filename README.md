## Music Genre Classifier

This repository intends to build a classification
method for song lyrics into its respective music genres.
It does so by building latent vectors with Latent
Dirichlet Allocation and with feedforward neural networks
as classifiers.

The repository holds Poetry as a virtual environment manager.

###Usage
Use ´´´poetry install´´´ to deploy the necessary dependencies.

#Training Latent Dirichlet Allocation for Topic Modelling Vectors

´´´´ python src\train_topic_modeller_script.py -t {Training Size} -n {Number of Topics}´´´´

Where:
Training Size: Size of data for topic extraction
Number of Topics: Amount of different topic classes to generate. 


#Training MLP Classifier

´´´´ python src\train_nn_script.py -t {Training Size}´´´´

Where:
Training Size: Size of data for neural network model.


