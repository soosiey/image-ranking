# image_ranking


## Instructions to run

The dataset for TinyImageNet must first be placed in a folder labeled 'data' outside of the repo. If the repo is cloned, then '../data/' must contain the dataset.

In main.py, change the model to a desired model, or use resnet50 as we have defaulted to. Then, simply run main.py to get a trained model saved to models/trained_models.
The last epoch, which is epoch 9, will have the final model.

You can then change start_epoch in test.py, to a epoch that has been saved, or use 9, as it is set as a default. Create a folder labeled 'embeddings', and then
run test.py to create the training and testing embeddings in the 'embeddings' folder. Running test.py will also print the test accuracy. The last printed line will have a tuple
with two numbers, (ex. (0.xxxx, 0.xxxx)). The first number in this tuple is the top-30 accuracy. The k for top-k accuracy can be changed in the test.py file, when calling data.knn_accuracy().

Finally, to get the top and bottom images, run grapher.py. Make sure that the model and start_epoch variables are the same as the previous steps. The images will be saved to a folder labeled 'images'.


To run everything as default, for example, you can simply run main.py, test.py, and grapher.py as is.

The utils.py file contains the Data class, custom dataloader, and all the functions used for training, obtaining embeddings, getting accuracies, and getting the top and bottom images.

val_precision.py and similarity_precision.py get the precisions for the datasets using the final model. Again, make sure the model is saved to models/trained_models after running main.py.
