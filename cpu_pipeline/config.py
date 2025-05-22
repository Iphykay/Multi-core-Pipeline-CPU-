module_name = "config"
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

# OTHER IMPORTS
from keras_tuner  import Objective
from mne.channels import make_standard_montage
from numpy        import pi, log10, array
from os           import cpu_count

# Main
#region
ConfigMain                           = {}
ConfigMain['classPrdt_gl']           = 'classPrdct'
ConfigMain['col_locEEG_gl']          = ['X','Y','Z']
ConfigMain['folder_info_gl']         = 'Info'
ConfigMain['folder_data_gl']         = 'Data'
ConfigMain['folder_hptLog_gl']       = 'HPTLOG'
ConfigMain['folder_prdct_gl']        = 'Predict'
ConfigMain['folder_prjct_gl']        = 'MINDPRINT'
ConfigMain['folder_model_gl']        = 'Model'
ConfigMain['folder_fusion_gl']       = 'Fusion'
ConfigMain['folder_modelPlot_gl']    = f'{ConfigMain['folder_model_gl']}\\Plot'    
ConfigMain['folder_perScores_gl']    = f'{ConfigMain['folder_prdct_gl']}\\Perf'   
ConfigMain['folder_perPlot_gl']      = f'{ConfigMain['folder_prdct_gl']}\\Plot' 
ConfigMain['folder_BstperScores_gl'] = f'{ConfigMain['folder_prdct_gl']}\\Perf\\Best'  
ConfigMain['folder_BstperPlot_gl']   = f'{ConfigMain['folder_prdct_gl']}\\BestPlot'
ConfigMain['numBestModels_gl']       = 1
ConfigMain['PI_DIV_2_GL']            = pi/2
ConfigMain['scoreIndvdl_gl']         = 'scoreIndvdl'
ConfigMain['scoreOvrall_gl']         = 'scoreOvrall'
ConfigMain['scorePrdct_gl']          = 'ScorePrdct'
ConfigMain['timeHrs_gl']             = 3600
ConfigMain['timeMins_gl']            = 60
ConfigMain['DataSetName_gl']         = ('Trn', 'Val', 'Tst') 
ConfigMain['keys_prdctPerf_gl']      = ('cLabel','prdctScore','perfScoreIndvdl','perfScoreOvrall')
ConfigMain['ExportSamplePlot_gl']    = 'SamplePlot'
ConfigMain['numWindows_gl']          = 3
ConfigMain['isPrintTest_gl']         = True
ConfigMain['timeinsecs_gl']          = 3600
ConfigMain['numCores_gl']            = cpu_count()
#endregion

# Export Sample
ConfigMain['streamAll_gl']           = 'streamAll'
ConfigMain['soiEEG_fltrd_Ntch_gl']   = 'soiEEG_fltrd_Ntch'
ConfigMain['soiEEG_fltrd_Imp_gl']    = 'soiEEG_fltrd_Imp'
ConfigMain['soiEEG_fltrd_BndPas_gl'] = 'soiEEG_fltrd_BndPas'
ConfigMain['soiFltrdBand_gl']        = 'soiFltrdBand'
ConfigMain['soiCndtn_gl']            = 'soiCndtn'
ConfigMain['soiAllPR_gl']            = 'soiAllPR'

# Statistical Feature Representation
#region
ConfigStatF                     = {}
ConfigStatF['colNamesStat_gl']  = ['M', 'S', 'K', 'A']
ConfigStatF['feature_gl']       = ['MM','SM','MS','SS','MK','SK','MA','SA']  # this will change
ConfigStatF['mainCols_mne_gl']  = ['Semester','Subjects','Sessions','Channels','Trial_ID','soI_ID']
ConfigStatF['numWdws_n0_gl']    = 15   # number of windows without overlap 
ConfigStatF['numCols_ss_gl']    = 8    # number of columns on the dfs for 2nd lvl stats
ConfigStatF['numWdwStats_gl']   = 8    # for 8 feats (stat_stats)
ConfigStatF['perNonOvrlap_gl']  = 0.50
#endregion

# Spatio-Temporal Representation
#region
ConfigST                 = {}
# SoI Duration: Length of SoI stream, Ratio of overlap in percent, Adjusted number in windows.
ConfigST['window_gl']    = {'k_adjstd_gl':{5: 32, 3: 32, 0.5: 32},
                            'ratio_Ovrlap':{5: 0.579102, 3: 0.334961, 0.5: 0.4570},
                            'window_width':{5: 128, 3: 128, 0.5: 16}}  # Must match LoadPrepare_ui[‘durationWindow_ui’] and LoadPrepare_ui['thSkipSoi_ui'] 
ConfigST['num_views_gl'] = 3
#endregion

# CONSTANTS
# Extract Soi
#region
ConfigExtrctSoi                     = {}
ConfigExtrctSoi['colmnGrpBy1st_gl'] = 'subject' 
ConfigExtrctSoi['colmnGrpBy2nd_gl'] = 'session'
ConfigExtrctSoi['NUM_CORDNATE_GL']  = 3 
ConfigExtrctSoi['numSoiDf_cols_gl'] = ['semester', 'subject', 'session', 'event_Type', 
                                       'event_code', 'trialID', 'eye_state']
ConfigExtrctSoi['rawEEG_file_gl']   = 'rawEEG'
#endregion

# Preprocessing
#region
ConfigPrePrcss                     = {}
ConfigPrePrcss['applyBandPass_gl'] = True
ConfigPrePrcss['applyNotch_gl']    = True
ConfigPrePrcss['applyImp_gl']      = True
ConfigPrePrcss['betaSftmin_gl']    = -0.2 
ConfigPrePrcss['freqsNotch_gl']    = [60,120,180,240]
ConfigPrePrcss['highFreq_gl']      = None
ConfigPrePrcss['lowerFreq_gl']     = 0.5
ConfigPrePrcss['num_ch_gl']        = 24 
ConfigPrePrcss['thrshld_gl']       = 10**-5
ConfigPrePrcss['test_size_gl']     = 0.2
#endregion

# FeatureGenerator_HC
#region
ConfigFeatGen_HC                 = {}
ConfigFeatGen_HC['bgn_col_gl']   = 6       # the start index to save the dictionary per channel of the windows
ConfigFeatGen_HC['kurt_srch_gl'] = array(['K'])
ConfigFeatGen_HC['mean_srch_gl'] = array(['M'])
ConfigFeatGen_HC['numCol_gl']    = 11      # number of columns on the dfs for 1st lvl stats
ConfigFeatGen_HC['std_srch_gl']  = array(['S'])
ConfigFeatGen_HC['skew_srch_gl'] = array(['A'])

# Plot
ConfigFeatGen_HC['col_x2_gl'] = 'Model'
ConfigFeatGen_HC['col_y_gl']  = 'Score'
#endregion

# Dcnn and Ae
#region
ConfigDcnnAe                           = {}
ConfigDcnnAe['LOG_10_2_GL']            = log10(2)
ConfigDcnnAe['ktObjective_gl']         = Objective('val_loss','min')
ConfigDcnnAe['NUM_LAYRS_UI']           = 2    # to get the last layer index
ConfigDcnnAe['SIZE_LAST_LAYER_IMG_UI'] = 2    # To avoid having the size of the grid data being odd 
#endregion

# Rc
#region
ConfigRc                      = {}
ConfigRc['ktObjective_gl']    = Objective('score','max')
ConfigRc['ktObjectiveAnn_gl'] = ConfigDcnnAe['ktObjective_gl']
#endregion

# This section is for EEG
#region
ConfigEEG                    = {}
ConfigEEG['AccChnls_gl']     = ("AccX", "AccY", "AccZ")
ConfigEEG['eeg_channels_gl'] = ("Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", 
                                "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz", "M1", "M2", "AFz", "CPz", "POz")
ConfigEEG['bandEEG_gl']      = {'entire': (1, None),
                                'delta': (1, 4),
                                'theta': (4, 8),
                                'alpha': (8, 13),
                                'betaa': (13, 32),
                                'gamma': (32, 125),
                                'highr': (125, None),
                                'th2Bt': (4,32)}
ConfigEEG['QuarChnls_gl']    = ("QuarX", "QuarY", "QuarZ", "QuarW")
ConfigEEG['GyroChnls_gl']    = ("GyroX", "GyroY", "GyroZ")
ConfigEEG['IMU_channels_gl'] = ConfigEEG['QuarChnls_gl'] + ConfigEEG['AccChnls_gl'] + ConfigEEG['GyroChnls_gl']
ConfigEEG['MBT_NumEEGCh']    = len(ConfigEEG['eeg_channels_gl'])
ConfigEEG['MBT_NumQuatCh']   = len(ConfigEEG['QuarChnls_gl'])
ConfigEEG['MBT_NumAccCh']    = len(ConfigEEG['AccChnls_gl'])
ConfigEEG['MBT_NumGyroCh']   = len(ConfigEEG['GyroChnls_gl'])
#endregion

#region 
ConfigEEG['TYPE_CH']     = ['eeg'] * ConfigEEG['MBT_NumEEGCh'] + ['ecg'] * ConfigEEG['MBT_NumAccCh'] +  ['ecog'] * ConfigEEG['MBT_NumGyroCh'] + ['ref_meg'] * ConfigEEG['MBT_NumQuatCh']      
ConfigEEG['MONTAGE_STD'] = make_standard_montage('standard_1020')
#endregion


# INITIALIZATION













