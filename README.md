## Music Genre Classifier

Classification of songs can help music distributers categorize
and group songs for recommendation systems. Often times songs
are classified with audio processing and computer vision to find
patterns in audiowaves. Nonetheless, it would be interesting to
find patterns in song lyrics to that indicate pertenance to 
specific or multiple music genres. This could potentially
serve as a feature to improve existing genre classification methods.

This proyect intends to build a classification
method for song lyrics into its respective music genres.
by building latent vectors with help of topic modelling
and with feedforward neural networks.

The repository holds Poetry as a virtual environment manager.

### Usage
Use ```poetry install``` to deploy the necessary dependencies.

## Training Latent Dirichlet Allocation

``` python src\train_topic_modeller_script.py -t {Training Size} -n {Number of Topics}```

Where:
`Training Size`: Size of data for topic extraction
Number of Topics: Amount of different topic classes to generate. 


## Training FFNN Classifier

``` python src\train_nn_script.py -t {Training Size}```

Where:
`Training Size`: Size of data for neural network model.


