module_name = 'ui'

'''
Version: v1.0.0

Description:
   All varaibles used for the entire pipeline

Authors:
    Dr. Omer Muhammet Soysal
    Iphy Kelvin

Date Created     : 12/23/2024
Date Last Updated: 12/23/2024

Doc:

Notes:
    <***>
'''

# OTHER IMPORTS 
# from Classifier       import KNN, RF, SVM, ANN   # not in use for now Issue Pending
# from FGenerator_ML    import dcnn                # not in use for now Issue Pending
from itertools        import combinations
from os               import environ
from sklearn.metrics  import accuracy_score, make_scorer
environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
environ['TF_CPP_MIN_LOG_LEVEL']  = '1'
import tensorflow      as tf

prfx_exp  = 'ST_' ; derive_expId  = True; expId = None; suffix_exp = ''

# All
#region
All_ui                         = {}
All_ui['eye_state_ui']         = 'EC' 
All_ui['event_code_ui']        = ['bk_pic'] 
All_ui['num_fold_ui']          = 3
All_ui['foldIDLst_ui']         = [f'F{i}' for i in range(All_ui['num_fold_ui'])]
All_ui['leftExtnt_ui']         = 0                            # Read rawEEG from leftExtnt samples before the begind time-stamp
All_ui['nameRCLst_ui']         = ('KNN','RF','SVM','ANN')     # 'KNN','RF','SVM','ANN'
All_ui['nameFGLst_ui']         = ('DCNN','AE') 
All_ui['pathSoiRoot_ui']       = '..\\DATA\\SOI_Sp23_u\\'
All_ui['rghtExtnt_ui']         = 0                            # to rghtExtnt samples after the end time-stamp of the SoI
All_ui['soiDfAll_file_ui']     = 'soiDfAll.pckl'
All_ui['sem_ui']               = ('Sp23',)                    # ('Fa23','Sp24')
All_ui['subjectsAll_ui']       = ('sb328','sb381','sb455','sb717','sb330','sb768','sb106')       #  
All_ui['se_ui']                = ('se1','se2')
All_ui['stmlusName_ui']        = f'{All_ui['event_code_ui'][0]}_{All_ui['eye_state_ui']}'
All_ui['sizeX_ui']             = 16
All_ui['sizeY_ui']             = 16
All_ui['trial_id_ui']          = (0,1,2,3,4,5,6,7,8,9,10)
All_ui['queryVal_ui']          = {'keyStmls'   : (),
                                  'key_soi'    : (),
                                  'event_Type' : (),
                                  'event_code' : All_ui['event_code_ui'],   
                                  'semester'   : All_ui['sem_ui'] ,
                                  'subject'    : All_ui['subjectsAll_ui'],  
                                  'session'    : All_ui['se_ui'],    
                                  'trialID'    : All_ui['trial_id_ui'],
                                  'trial_group': (),
                                  'marker_ID'  : (),
                                  'eye_state'  : (All_ui['eye_state_ui'],),
                                  'time_begin' : (),
                                  'time_end'   : (),
                                  'begin_indx' : (),
                                  'end_indx'   : (),
                                  'source_file': ()}
All_ui['cSubjects_ui']         = {key: value for value, key in enumerate(All_ui['subjectsAll_ui'])}   # subjects
All_ui['cLabels_ui']           = {value: key for value, key in enumerate(All_ui['subjectsAll_ui'])}   # class labels for each subject
All_ui['isFGRC_ui']            = True
All_ui['isExport_ui']          = True
All_ui['nameFGRCLst_ui']       = ('DCNN_KNN','DCNN_RF','DCNN_SVM','DCNN_ANN','AE_KNN','AE_RF','AE_SVM','AE_ANN')
All_ui['loadpreprcss_ui']      = False
All_ui['numSbjsAll_ui']        = len(All_ui['subjectsAll_ui'])
All_ui['numSbjsCmd_ui']        = 2    # For Binary: 2
All_ui['combntn_ui']           = list(combinations(All_ui['subjectsAll_ui'],All_ui['numSbjsCmd_ui']))
All_ui['get_feats_ui']         = True
All_ui['strategy_ui']          = tf.distribute.MirroredStrategy()
#endregion

# LoadPrepareInput
#region
# Export SoI and View-Matrix
LoadPrepare_ui                      = {}
LoadPrepare_ui['bandToUse_ui']      = ['entire'] # ,'delta','theta','alpha','betaa','gamma'
LoadPrepare_ui['useStdMontage_ui']  = False
LoadPrepare_ui['RANDOM_STATE_GL']   = 15
LoadPrepare_ui['useViewModel_ui']   = False
LoadPrepare_ui['incldMotion_ui']    = True
LoadPrepare_ui['num_soi2Use_ui']    = None # 20
LoadPrepare_ui['querySpecial_ui']   = False        # for bk_pic, eye_state = {EO,EC}
LoadPrepare_ui['thSkipSoi_ui']      = [0.45, 0.52] #  EC-1Sec: [0.45, 0.52]; EC:[2.9, 5.1] EO:[0.75, 5.1]; 3 Sec: [2.9, 3.1] #Threshold to skip an SoI, in seconds [0.45, 0.52] -> [0.75, 5.1]
LoadPrepare_ui['durationWindow_ui'] = 0.5          # Duration of a window in seconds
LoadPrepare_ui['thrshldImp_ui']     = 20 
LoadPrepare_ui['viewIDLst_ui']      = ['1','2','3']      # for single or multiple.
LoadPrepare_ui['ViewUsed_ui']       = '1'

# Filter
LoadPrepare_ui['applyArtifact_ui'] = False
LoadPrepare_ui['applyRef_ui']      = False

# Pre-process
LoadPrepare_ui['thrshSkip_ui']     = 50         # Number of samples to disgard an SoI shorter than expected.
LoadPrepare_ui['exportPP_ui']      = True
LoadPrepare_ui['exportPR_ui']      = True

# Pre-representation
LoadPrepare_ui['useSoftMin_ui']    = False
LoadPrepare_ui['useInv_ui']        = False
LoadPrepare_ui['useLOM_ui']        = False
LoadPrepare_ui['useLOG_ui']        = False
LoadPrepare_ui['preRepFnctn']      = 'None'       # Smi (Softmin), Inv (Inverse), None
LoadPrepare_ui['sgnalConFnctn']    = 'Nan'      # 'LoG',     'LoM',     'Nan'
LoadPrepare_ui['fNamePP']          = f'{LoadPrepare_ui['sgnalConFnctn']}_{LoadPrepare_ui['preRepFnctn']}'
#endregion

# Data Split and Shuffle
#region
SplitShufl                = {}          # SS: SplitShuffle
SplitShufl['main_str_ui'] = 'session'  #'Semesters'; 'Sessions'
SplitShufl['trn_str_ui']  = ['se1']     # ['Fa23'], ['se1']
SplitShufl['tst_str_ui']  = ['se2']     # ['Sp24']; ['se2']
SplitShufl['seed_num_ui'] = 20
#endregion

# DCNN
#region
Dcnn_ui                              = {}
num_init_pts_cnv2D_ui                = {}

# HYPERPARAMETER
Dcnn_ui['actvtn_lst_cnv2D_ui']       = ['elu','selu']
Dcnn_ui['btchsze_min_cnv2D_ui']      = 32
Dcnn_ui['btchsze_max_cnv2D_ui']      = 128
Dcnn_ui['btchsze_stp_cnv2D_ui']      = 32
Dcnn_ui['drpout_lst_cnv2D_ui']       = [0.2,0.4,0.6]
Dcnn_ui['exetns_per_trial_cnv2D_ui'] = 1
Dcnn_ui['kernel_cnv2D_ui']           = 3
Dcnn_ui['loss_cnv2D_ui']             = 'sparse_categorical_crossentropy'
Dcnn_ui['min_value_cnv2D_ui']        = 10
Dcnn_ui['max_value_cnv2D_ui']        = 200
Dcnn_ui['min_lyr_cnv2D_ui']          = 2
Dcnn_ui['min_epch_cnv2D_ui']         = 100 
Dcnn_ui['max_epch_cnv2D_ui']         = 300
Dcnn_ui['max_trials_cnv2D_ui']       = 1
Dcnn_ui['optmzr_lst_cnv2D_ui']       = ['adam','adamW']
Dcnn_ui['pool_cnv2D_ui']             = 2
Dcnn_ui['step_cnv2D_ui']             = 10
Dcnn_ui['step_epch_cnv2D_ui']        = 50
Dcnn_ui['train_size_ui']             = 0.80
Dcnn_ui['val_size_ui']               = 0.20
Dcnn_ui['wght_lst_cnv2D_ui']         = ['glorot_uniform','he_uniform']
num_init_pts_cnv2D_ui['DCNN']        = 1612

# OTHER PARAMETERS
Dcnn_ui['metric_compile_cnv2D_ui'] = 'accuracy'
Dcnn_ui['model_name_ui']           = 'DCNN'
Dcnn_ui['perMeasurLst_ui']         = ['auc','f1','acc','rec','spe','pre','dor','adjF','matC',
                                      'kap','micF','balAcc','micAUC','micGeoMean','confMtrx']
Dcnn_ui['verbose_cnv2D_ui']        = 0

# PLOTTING
Dcnn_ui['fontSize_title_ui'] = 16
Dcnn_ui['fontSize_xy_ui']    = 12
Dcnn_ui['fontSize_lgnd_ui']  = 10
Dcnn_ui['figSize_barplt_ui'] = (12, 6)

# Violin Plot
Dcnn_ui['rw_subplt_ui']     = 2
Dcnn_ui['col_subplt_ui']    = 2
#endregion

# AE
#region
Ae_ui                           = {}
num_init_pts_ae_ui              = {}

# HYPERPARAMETER
Ae_ui['actvtn_lst_ae_ui']       = Dcnn_ui['actvtn_lst_cnv2D_ui']
Ae_ui['btchsze_min_ae_ui']      = Dcnn_ui['btchsze_min_cnv2D_ui']
Ae_ui['btchsze_max_ae_ui']      = Dcnn_ui['btchsze_max_cnv2D_ui']
Ae_ui['btchsze_stp_ae_ui']      = Dcnn_ui['btchsze_stp_cnv2D_ui']
Ae_ui['drpout_lst_ae_ui']       = Dcnn_ui['drpout_lst_cnv2D_ui'] 
Ae_ui['exetns_per_trial_ae_ui'] = Dcnn_ui['exetns_per_trial_cnv2D_ui']
Ae_ui['kernel_ae_ui']           = Dcnn_ui['kernel_cnv2D_ui']
Ae_ui['loss_ae_ui']             = 'mean_squared_error'
Ae_ui['min_value_ae_ui']        = Dcnn_ui['min_value_cnv2D_ui']
Ae_ui['max_value_ae_ui']        = Dcnn_ui['max_value_cnv2D_ui']
Ae_ui['min_lyr_ae_ui']          = Dcnn_ui['min_lyr_cnv2D_ui']
Ae_ui['min_epch_ae_ui']         = Dcnn_ui['min_epch_cnv2D_ui'] 
Ae_ui['max_epch_ae_ui']         = Dcnn_ui['max_epch_cnv2D_ui']
Ae_ui['max_trials_ae_ui']       = Dcnn_ui['max_trials_cnv2D_ui'] 
Ae_ui['optmzr_lst_ae_ui']       = Dcnn_ui['optmzr_lst_cnv2D_ui']
Ae_ui['pool_ae_ui']             = Dcnn_ui['pool_cnv2D_ui']
Ae_ui['step_ae_ui']             = Dcnn_ui['step_cnv2D_ui']
Ae_ui['step_epch_ae_ui']        = Dcnn_ui['step_epch_cnv2D_ui']
Ae_ui['train_size_ui']          = 0.80
Ae_ui['val_size_ui']            = 0.20
Ae_ui['wght_lst_ae_ui']         = Dcnn_ui['wght_lst_cnv2D_ui']
Ae_ui['verbose_ae_ui']          = Dcnn_ui['verbose_cnv2D_ui']
num_init_pts_ae_ui['AE']        = 1612

# OTHER PARAMETERS
Ae_ui['metric_compile_ui'] = 'accuracy'
Ae_ui['model_name_ui']     = 'AE'
Ae_ui['perMeasurLst_ui']   = ['auc','f1','acc','rec','spe','pre','dor','adjF','matC',
                              'kap','micF','balAcc','micAUC','micGeoMean','confMtrx']
Ae_ui['verbose_cnv2D_ui']  = 0

# PLOTTING
Ae_ui['fontSize_title_ui'] = 16
Ae_ui['fontSize_xy_ui']    = 12
Ae_ui['fontSize_lgnd_ui']  = 10
Ae_ui['figSize_barplt_ui'] = (12, 6)

# Violin Plot
Ae_ui['rw_subplt_ui']     = 2
Ae_ui['col_subplt_ui']    = 2

# CONSTANTS
Ae_ui['NUM_LAYRS_UI']           = 2    # to get the last layer index
Ae_ui['SIZE_LAST_LAYER_IMG_UI'] = 2    # To avoid having the size of the grid data being odd
#endregion

# FG+RC
#region
Fgrc_ui                           = {}
num_init_pts_fgrc_ui              = {}
Fgrc_ui['model_name_DCNN_KNN_ui'] = 'DCNN_KNN'
Fgrc_ui['model_name_DCNN_SVM_ui'] = 'DCNN_SVM'  
Fgrc_ui['model_name_DCNN_RF_ui']  = 'DCNN_RF'           
Fgrc_ui['model_name_DCNN_ANN_ui'] = 'DCNN_ANN'  
Fgrc_ui['model_name_AE_KNN_ui']   = 'AE_KNN'  
Fgrc_ui['model_name_AE_SVM_ui']   = 'AE_SVM'  
Fgrc_ui['model_name_AE_RF_ui']    = 'AE_RF'  
Fgrc_ui['model_name_AE_ANN_ui']   = 'AE_ANN' 
num_init_pts_fgrc_ui['DCNN_ANN']  = 1612
num_init_pts_fgrc_ui['DCNN_KNN']  = 48
num_init_pts_fgrc_ui['DCNN_RF']   = 48
num_init_pts_fgrc_ui['DCNN_SVM']  = 36 
num_init_pts_fgrc_ui['AE_ANN']    = 1612
num_init_pts_fgrc_ui['AE_KNN']    = 48
num_init_pts_fgrc_ui['AE_RF']     = 48
num_init_pts_fgrc_ui['AE_SVM']    = 36 
#endregion

# Regular Classifers
num_init_pts_rc_ui           = {}

#region
Rc_ui                        = {}
Rc_ui['fontSize_title_ui']   = 16
Rc_ui['fontSize_xy_ui']      = 12
Rc_ui['fontSize_lgnd_ui']    = 10
Rc_ui['figSize_barplt_ui']   = (12, 6)
Rc_ui['max_trials_ui']       = 1  
Rc_ui['isRc_ui']             = True
Rc_ui['isBinary_ui']         = True   # For Binary: True
Rc_ui['perMeasurLst_ui']     = Dcnn_ui['perMeasurLst_ui']
Rc_ui['scoring_ui']          = make_scorer(accuracy_score)

# Violin Plot
Rc_ui['rw_subplt_ui']     = 2
Rc_ui['col_subplt_ui']    = 2
#endregion

# ANN
#region
Ann_ui                            = {}
Ann_ui['actvtn_lst_ann_ui']       = Dcnn_ui['actvtn_lst_cnv2D_ui']
Ann_ui['bias_value_ann_ui']       = 0.01
Ann_ui['drpout_lst_ann_ui']       = Dcnn_ui['drpout_lst_cnv2D_ui']
Ann_ui['exetns_per_trial_ann_ui'] = Dcnn_ui['exetns_per_trial_cnv2D_ui']
Ann_ui['min_value_ann_ui']        = 50 
Ann_ui['max_value_ann_ui']        = 200 
Ann_ui['min_lyr_ann_ui']          = 3
Ann_ui['model_name_ui']           = 'ANN'
Ann_ui['max_lyr_ann_ui']          = 5
Ann_ui['max_trials_ann_ui']       = Dcnn_ui['max_trials_cnv2D_ui']
Ann_ui['metric_compile_ui']       = Dcnn_ui['metric_compile_cnv2D_ui']
Ann_ui['optmzr_lst_ann_ui']       = Dcnn_ui['optmzr_lst_cnv2D_ui']
Ann_ui['step_ann_ui']             = 25 
Ann_ui['wght_lst_ann_ui']         = Dcnn_ui['wght_lst_cnv2D_ui']
Ann_ui['verbose_cnv2D_ui']        = Dcnn_ui['verbose_cnv2D_ui']
num_init_pts_rc_ui['ANN']         = 1612
#endregion

# KNN
#region
Knn_ui                    = {}
Knn_ui['knnwgt_lst_ui']   = ['uniform','distance']
Knn_ui['knnmtrc_lst_ui']  = ['cityblock','euclidean']
Knn_ui['lfmin_ui']        = 20
Knn_ui['lfmax_ui']        = 40
Knn_ui['model_name_ui']   = 'KNN'
Knn_ui['nghbrmin_ui']     = 3
Knn_ui['nghbrmax_ui']     = 5  # 12
num_init_pts_rc_ui['KNN'] = 48
#endregion

# RF
#region
Rf_ui                     = {}
Rf_ui['model_name_ui']    = 'RF'
Rf_ui['rfMxdpthMin_ui']   = 5
Rf_ui['rfMxdpthMax_ui']   = 15
Rf_ui['rfMinSplitMin_ui'] = 2
Rf_ui['rfMinSplitMax_ui'] = 4
Rf_ui['rfCritLst_ui']     = ['gini','entropy']
Rf_ui['rfEstMin_ui']      = 100
Rf_ui['rfEstMax_ui']      = 301
Rf_ui['rfEstStp_ui']      = 100
Rf_ui['maxFeatures_ui']   = 8
num_init_pts_rc_ui['RF']  = 48
#endregion

# SVM
#region
Svm_ui                    = {}
Svm_ui['model_name_ui']   = 'SVM'
Svm_ui['svm_cLst']        = [0.1,1.0,10.0,100.0]
Svm_ui['svm_gamaLst']     = [0.1,1.0]
Svm_ui['svm_krnlLst']     = ['linear','rbf','poly']
num_init_pts_rc_ui['SVM'] = 36
#endregion

# Check and Derive Parameters
#region
if LoadPrepare_ui['thSkipSoi_ui'][0]>LoadPrepare_ui['durationWindow_ui'] or LoadPrepare_ui['durationWindow_ui'] > LoadPrepare_ui['thSkipSoi_ui'][1]:
    print(f'Time Sample is not within the given section {LoadPrepare_ui['thSkipSoi_ui'][0]} and {LoadPrepare_ui['thSkipSoi_ui'][1]}', flush=True)
    exit()
# if

#region
if SplitShufl['main_str_ui'] == 'session':
    if (SplitShufl['trn_str_ui'][0] and SplitShufl['tst_str_ui'][0]) not in All_ui['se_ui']:
        print(f'{SplitShufl['trn_str_ui'][0]} and {SplitShufl['tst_str_ui'][0]} must be in {All_ui['se_ui']}', flush=True)
        exit()
    # if
else:
    if (SplitShufl['trn_str_ui'][0] and SplitShufl['tst_str_ui'][0]) not in All_ui['sem_ui']:
        print(f'{SplitShufl['trn_str_ui'][0]} and {SplitShufl['tst_str_ui'][0]} must be in {All_ui['sem_ui']}', flush=True)
        exit()
    # if
# if
#endregion

#region
if derive_expId:
    if  expId is None:
        if LoadPrepare_ui['preRepFnctn'] is None: preRepFnctn = ''
        else: preRepFnctn = LoadPrepare_ui['preRepFnctn']
        if LoadPrepare_ui['useViewModel_ui'] is False: nameRep_gl = f'\\StatsRepr'
        else: nameRep_gl = f'\\View{'_'.join(LoadPrepare_ui['viewIDLst_ui'])}' if LoadPrepare_ui['viewIDLst_ui'] else ''
        All_ui['expId'] = f'{prfx_exp}{LoadPrepare_ui['sgnalConFnctn']}{preRepFnctn}\\{All_ui['stmlusName_ui'].lower()}{All_ui['numSbjsCmd_ui']}s{Dcnn_ui['max_trials_cnv2D_ui']}t{suffix_exp}{nameRep_gl}'
    else:
        print(f'{derive_expId} and {expId} conflict!', flush=True)
        exit()
    # if
elif expId is None:
    print(f'{derive_expId} and {expId} conflict!',flush=True)
    exit()
# if

All_ui['pathOutput_ui']  = f'..\\Output\\{All_ui['expId']}'
All_ui['pathInput_ui']   = f'..\\Input\\{All_ui['expId']}'
All_ui['pathElapsed_ui'] = All_ui['pathOutput_ui']
All_ui['soiAllFilename'] = {'soiAllPR': f'{LoadPrepare_ui['fNamePP']}.pckl',  'soiAllFltrBand': f'{LoadPrepare_ui['fNamePP']}_Band.pckl'}
#endregion

#region
knn_param_grid = dict(n_neighbors = list(range(Knn_ui['nghbrmin_ui'], Knn_ui['nghbrmax_ui'])), 
                      leaf_size = list(range(Knn_ui['lfmin_ui'],Knn_ui['lfmax_ui'])),
                      weights = Knn_ui['knnwgt_lst_ui'],metric = Knn_ui['knnmtrc_lst_ui'])

svm_param_grid = {'C':Svm_ui['svm_cLst'], 'gamma':Svm_ui['svm_gamaLst'], 'kernel':Svm_ui['svm_krnlLst']} 

rndfrst_param_grid = {'n_estimators':list(range(Rf_ui['rfEstMin_ui'], Rf_ui['rfEstMax_ui'], Rf_ui['rfEstStp_ui'])),
                      'criterion':Rf_ui['rfCritLst_ui'],
                      'max_depth':list(range(Rf_ui['rfMxdpthMin_ui'],Rf_ui['rfMxdpthMax_ui'])),
                      'min_samples_split': list(range(Rf_ui['rfMinSplitMin_ui'], Rf_ui['rfMinSplitMax_ui'])),
                      'max_features':[Rf_ui['maxFeatures_ui']]}
ann_param = str({'activation':Ann_ui['actvtn_lst_ann_ui'],'Dropout':Ann_ui['drpout_lst_ann_ui'] ,'optimizer':Ann_ui['optmzr_lst_ann_ui'] , 
                 'kernel_initializer':Ann_ui['wght_lst_ann_ui'],'ann_layers':range(3,5),'batchsize':range(32,128,4),
                 'epochs':range(100,200,50),'units':range(10,200,20)})
cnn_param = str({'activation':Ann_ui['actvtn_lst_ann_ui'], 'Dropout':Ann_ui['drpout_lst_ann_ui'], 'optimizer':Ann_ui['optmzr_lst_ann_ui'] ,
                 'kernel_initializer':Ann_ui['wght_lst_ann_ui'], 'pooling':[2], 'kernel':[3],'cnn_layers':range(1,3),
                 'ann_layers':range(3,5),'units':range(10,200,20),'batchsize':range(32,128,4),
                 'epochs':range(100,200,50),'cnn_filters':range(10,200,4)})
dcnn_knn_param = {'DCNN':cnn_param,'KNN':str(knn_param_grid)}
dcnn_svm_param = {'DCNN':cnn_param,'SVM':str(svm_param_grid)}
dcnn_rdf_param = {'DCNN':cnn_param,'RDFRST':str(rndfrst_param_grid)}
dcnn_ann_param = {'DCNN':cnn_param,'ANN':ann_param}
ae_knn_param   = {'AE':cnn_param,'KNN':str(knn_param_grid)}
ae_svm_param   = {'AE':cnn_param,'SVM':str(svm_param_grid)}
ae_rdf_param   = {'AE':cnn_param,'RDFRST':str(rndfrst_param_grid)}
ae_ann_param   = {'AE':cnn_param, 'ANN':ann_param}
allmodelsprm   = {'KNN':str(knn_param_grid),'SVM':str(svm_param_grid),'RDFRST':str(rndfrst_param_grid),'ANN':ann_param,
                  'DCNN_KNN':dcnn_knn_param,'DCNN_SVM':dcnn_svm_param,'DCNN_RDFRST':dcnn_rdf_param,'DCNN_ANN':dcnn_ann_param,
                  'AE_KNN':ae_knn_param,'AE_SVM':ae_svm_param,'AE_RDFRST':ae_rdf_param,'AE_ANN':ae_ann_param}
#endregion