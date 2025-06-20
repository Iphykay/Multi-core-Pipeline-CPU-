# Multi-core-Pipeline-CPU
The pipeline uses multiprocessing module in python to facilitate training, tuning and 
analyzing brain signals.

The pipeline is an end-to-end, that takes the brain signal, applies preprocessing and 
converts the 1D time series to image data. 

The brain signals are loaded in the:

    loader.py

using subject, session, semester and stimulus name. The preprocessing stage occurs 
in the same module.

The machine learning models and deep learning models are in:

    network_rcs.py and rc_models.py

The Deep learning models are trained using the image data and features extracted 
from the last pooling layers.  These features are used in training regular classifiers 
(Random Forest, ANN, SVM and KNN) to identify participants.

To run the process:

        use main.py

The config files hold variables used in the end-to-end pipeline.

An update to this pipeline can not be made public due to signed NDA.
