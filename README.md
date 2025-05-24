# Multi-core-Pipeline-CPU-
The pipeline uses multiprocessing module in python to facilitate training, tuning and analyzing brain signals.

The pipeline is an end-to-end pipeline, that takes the brain signal, applies preprocessing and converts the 1D time series
to image data. 

The Deep learning models are trained using the image data and features extracted from the last pooling layers. 
These features are used in training regular classifiers (Random Forest, ANN, SVM and KNN) to identify participants.

An update to this pipeline can not be made public due to signed NDA.
