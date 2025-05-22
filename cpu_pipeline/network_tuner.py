module_name = 'hptune'

'''
Version: v1.0.0

Description:
   Modifies hyperparameters

Authors:
    Dr. Omer Muhammet Soysal
    Iphy Kelvin

Date Created     : 02/28/2025
Date Last Updated: 02/28/2025

Doc:

Notes:
    <***>
'''


# CUSTOM IMPORTS
from ui import Dcnn_ui

# OTHER IMPORTS
from keras_tuner.tuners import BayesianOptimization

# USER INTERFACE
# FOR HPT
btchsze_min_cnv2D_ui = Dcnn_ui['btchsze_min_cnv2D_ui']
btchsze_max_cnv2D_ui = Dcnn_ui['btchsze_max_cnv2D_ui']
btchsze_stp_cnv2D_ui = Dcnn_ui['btchsze_stp_cnv2D_ui']
min_epch_cnv2D_ui    = Dcnn_ui['min_epch_cnv2D_ui']
max_epch_cnv2D_ui    = Dcnn_ui['max_epch_cnv2D_ui']
stp_epch_cnv2D_ui    = Dcnn_ui['step_epch_cnv2D_ui']

class customTuner(BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):

        # You can add additional Hyperparameters for preprocessing and custom training loops
        # via overriding 'run_trial'

        kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', btchsze_min_cnv2D_ui, btchsze_max_cnv2D_ui, 
                                                         step = btchsze_stp_cnv2D_ui)
        kwargs['epochs']     = trial.hyperparameters.Int('epochs', min_epch_cnv2D_ui, max_epch_cnv2D_ui, step=stp_epch_cnv2D_ui)
        return super(customTuner, self).run_trial(trial, *args, **kwargs)
    #

