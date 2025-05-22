module_name = "config_compile"
'''
Version: v1.0.0

Description:
   This module includes project confgurations and global variables.

Date Created     : 02/25/2025
Date Last Updated: 02/25/2025

Doc:
    <***>

Notes:
    <***>
'''


# CUSTOM IMPORTS
from ui import All_ui

# OTHER IMPORTS
from numpy  import min as npMin, mean as npMean, median as npMedian, max as npMax, std as npStd
from pandas import StringDtype, Float32Dtype


# CONSTANTS
#region
ConfigMainCmp                      = {}
ConfigMainCmp['modelRC_gl']        = ['KNN','RF','SVM', 'ANN']
ConfigMainCmp['modelDCNN_gl']      = ['DCNN_KNN','DCNN_RF','DCNN_SVM','DCNN_ANN']
ConfigMainCmp['modelAE_gl']        = ['AE_KNN','AE_RF','AE_SVM','AE_ANN'] 
ConfigMainCmp['RepType_gl']        = ['RC', 'DCNN','AE']                                      # Representation type
ConfigMainCmp['pmName_IPM_gl']     = ['AUC','F1','Accuracy','Recall','Specificity','Precision','DOR','AdjustedF']  # PM for each subject
ConfigMainCmp['pmName_OPM_gl']     = ['MattCorr','Kappa','MicroAUC','BalancedAcc','MicroF1','micGeoMean']     # PM out of all subjects
ConfigMainCmp['num_fold_gl']       = 3
ConfigMainCmp['numSbjsAll_ui']     = All_ui['numSbjsCmd_ui']
ConfigMainCmp['colNames_fold_gl']  = ['RepType','Group','Model','PM','Fold','Subject','Score']
ConfigMainCmp['colName_single_gl'] = ['RepType','Model','Subject','Group','IPM','Score']
ConfigMainCmp['colName_overal_gl'] = ['RepType','Model','Group','OPM','Score']
ConfigMainCmp['dType_sngl_gl']     = {'RepType':StringDtype(),'Model':StringDtype(), 'Subject':StringDtype(), 
                                      'Group':StringDtype(), 'IPM':StringDtype(), 'Score':Float32Dtype()}
ConfigMainCmp['dType_ovr_gl']      = {'RepType':StringDtype(),'Model':StringDtype(), 'Group':StringDtype(), 
                                      'OPM':StringDtype(), 'Score':Float32Dtype()}
ConfigMainCmp['functnStat_gl']     = [npMin, npMean, npMedian, npMax, npStd]
ConfigMainCmp['nameStat_gl']       = ['Min', 'Mean', 'Median', 'Max', 'Std']
ConfigMainCmp['folderStat_gl']     = ['Level_1']
ConfigMainCmp['rw_subplt_gl']      = 2
ConfigMainCmp['col_subplt_gl']     = 2
ConfigMainCmp['col_x1_gl']         = 'Subject'
ConfigMainCmp['col_x2_gl']         = 'Model'
ConfigMainCmp['col_y_gl']          = 'Score'
ConfigMainCmp['folderExprt_gl']    = 'Plot_Score'
#endregion

# MODELS
#region
ConfigModelsCmp = {}
ConfigModelsCmp['models_gl']           = {'RC': ConfigMainCmp['modelRC_gl'], 'DCNN': ConfigMainCmp['modelDCNN_gl'], 
                                          'AE': ConfigMainCmp['modelAE_gl']}
ConfigModelsCmp['knn_vrtns']           = ['KNN','DCNN_KNN','AE_KNN']
ConfigModelsCmp['svm_vrtns']           = ['SVM','DCNN_SVM','AE_SVM']
ConfigModelsCmp['rdfrst_vrtns']        = ['RDFRST','DCNN_RDFRST','AE_RDFRST'] 
ConfigModelsCmp['ann_vtrns']           = ['ANN','DCNN_ANN','AE_ANN']
ConfigModelsCmp['modelAllRc_gl']       = [ConfigModelsCmp['knn_vrtns'], ConfigModelsCmp['svm_vrtns'], 
                                          ConfigModelsCmp['rdfrst_vrtns'], ConfigModelsCmp['ann_vtrns']]
ConfigModelsCmp['modelNameDsplyRc_gl'] = ['KNN','SVM','RDFRST','ANN']
ConfigModelsCmp['modelAllRep_gl']      = [ConfigMainCmp['modelRC_gl'],ConfigMainCmp['modelDCNN_gl'],ConfigMainCmp['modelAE_gl']]

# Performance Measures
ConfigPMCmp = {}
ConfigPMCmp['pmName_gl'] = {'IPM': ConfigMainCmp['pmName_IPM_gl'], 'OPM': ConfigMainCmp['pmName_OPM_gl'], 
                            'ALL': ConfigMainCmp['pmName_IPM_gl'] + ConfigMainCmp['pmName_OPM_gl']}


# DERIVATIONS
ConfigMainCmp['numModels_gl'] = len(ConfigMainCmp['modelRC_gl']) + len(ConfigMainCmp['modelDCNN_gl']) + len(ConfigMainCmp['modelAE_gl'])
