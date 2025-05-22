module_name = 'FeatureGenerator_HC'

'''
Version: v1.0.0

Description:
   Read and preprocesses the streams 

Authors:
    Dr. Omer Muhammet Soysal
    Iphy Kelvin

Date Created     : 1/02/2025
Date Last Updated: 1/02/2025

Doc:

Notes:
    <***>
'''

# CUSTOM IMPORTS
from config   import ConfigStatF as CfgSF, ConfigPrePrcss as CfgPrePrcss, ConfigFeatGen_HC as CfgFeatGen, ConfigEEG as CfgEEG
from ui       import All_ui, LoadPrepare_ui as LdPrep_ui

# OTHER IMPORTS
from numpy        import zeros, array, mean, std, unique
from os           import makedirs as osMakedirs
from pickle       import dump
from pandas       import DataFrame
from scipy.stats  import skew, kurtosis

# USER INTERFACE
bandToUse_ui     = LdPrep_ui['bandToUse_ui']
cSubjects_ui     = All_ui['cSubjects_ui']
colNamesStat_ui  = CfgSF['colNamesStat_gl']
eyeState_ui      = All_ui['eye_state_ui']
mainCol_name_ui  = CfgSF['mainCols_mne_gl']
num_ch_ui        = CfgPrePrcss['num_ch_gl']
num_windws_n0_ui = CfgSF['numWdws_n0_gl']
numCols_ss_ui    = CfgSF['numCols_ss_gl']
numWdwStats_ui   = CfgSF['numWdwStats_gl']
per_nonOvrlap_ui = CfgSF['perNonOvrlap_gl']
pathInput_ui     = All_ui['pathInput_ui']
pathRootSoi_ui   = All_ui['pathSoiRoot_ui']

# COSTANTS
feature_gl       = CfgSF['feature_gl'] 

# CONFIGURATIONS
allTempDF_cols_gl = mainCol_name_ui + ['Window-ID'] + colNamesStat_ui
bgn_col_gl        = CfgFeatGen['bgn_col_gl']
kurt_srch_gl      = CfgFeatGen['kurt_srch_gl']
mean_srch_gl      = CfgFeatGen['mean_srch_gl']
numWdw_gl         = int(num_windws_n0_ui/per_nonOvrlap_ui) + 1 # adding +1 because of the index
numCol_gl         = CfgFeatGen['numCol_gl']
num_rtempDf_ss_gl = numWdwStats_ui*num_ch_ui                   # num of row per soi in statstats
ssCol_gl          = mainCol_name_ui + feature_gl
std_srch_gl       = CfgFeatGen['std_srch_gl']
skew_srch_gl      = CfgFeatGen['skew_srch_gl']
tempDf_ss_cols_gl = mainCol_name_ui + ['stat_type'] + ['statstat_values'] 
toKeepCol_gl      = mainCol_name_ui + feature_gl + ['class_label']


def getwindow_stats(window_width, data, len_nonOvrlap):
    '''
    Uses the shifting window to get statistical features of a signal.
    It calculates the mean, standard deviation, kurtosis and skewness in each window.

    Input:
    -----
    window_width : the size of the window width
    data         : input data  (vector 1d len_Soi)
    len_nonOvrlap: the overlap duration

    Output:
    -------  
    Return: an array for mean, std, kurtosis and skew
    '''
    FeatWndw = dict()

    for i_w in range(0, numWdw_gl):
        i_bgn = i_w*len_nonOvrlap
        data_windw = data[i_bgn:(i_bgn + window_width)]

        #Statistical features
        mean_ = mean(data_windw);  std_ = std(data_windw)
        kurt  = kurtosis(data_windw); skew_ = skew(data_windw)

        FeatWndw[f'Window_{i_w}'] = array([mean_, std_, kurt, skew_])
    # for

    return FeatWndw
#

def transform_data(features):
    '''
    Reshape features into [num_Soi, num_features, num_channels] per sm, sb, se
    num_Soi: Number of feature vectors
    
    Input:
    ------
    features: all the features [num_Soi*num_channels, num_feats]

    Output:
    ------
    : Returns transformed data   #[num_Soi, num_feats, num_channels] 
    '''          

    # size of the transformed data
    sizeTrnfrmd = features.shape[0]//num_ch_ui 

    # saving the class labels
    classLabel = zeros(shape=(sizeTrnfrmd,), dtype=int)

    # Session list
    sessionList = zeros(shape=(sizeTrnfrmd,), dtype=object)

    # Get the semester and subjects
    sem  = zeros(shape=(sizeTrnfrmd,), dtype=object)
    sbjt = zeros(shape=(sizeTrnfrmd,), dtype=object)
        
    # filter by soI_ID
    unique_soi = unique(features['soI_ID'].to_numpy())
            
    # the unique sois
    for i_soi in range(len(unique_soi)):
        sem[i_soi]         = features.loc[features['soI_ID'].isin([unique_soi[i_soi]])].loc[:,'Semester'].unique()[0]
        sbjt[i_soi]        = features.loc[features['soI_ID'].isin([unique_soi[i_soi]])].loc[:,'Subjects'].unique()[0]
        classLabel[i_soi]  = int(features.loc[features['soI_ID'].isin([unique_soi[i_soi]])].loc[:,'class_label'].unique()[0])
        sessionList[i_soi] = features.loc[features['soI_ID'].isin([unique_soi[i_soi]])].loc[:,'Sessions'].unique()[0]
    # for

    # Put the session and class label into a dataframe
    dataset_info = DataFrame({'semester': sem, 'subject': sbjt, 'session': sessionList,'soI_ID':unique_soi,
                              'class_labels': classLabel}, columns=['semester','subject','session','soI_ID','class_labels']) 
                
    # Reshaped the data into [num_Soi, num_feats, num_channels]
    df_trnsfrmd = features.loc[:,feature_gl]
    df_trnsfrmd = df_trnsfrmd.to_numpy().reshape(sizeTrnfrmd, len(feature_gl), num_ch_ui)
        
    return df_trnsfrmd, dataset_info                       
#

def save_data(data, path, pathpckl, name_kS=None, sb=None, se=None, name=None):
    '''
    Saves the 1st level stats and 2nd stats level data 
    
    Input:
    -----
    data    : data to be saved (Dataframe)
    path    : path where the data is to be saved
    pathpckl: pickle path where the data is to be saved
    keyStmls: keyStmls id
    name_ks : name of thr stimulus, str
    sb      : subject number, str (subject 204-sb204)
    se      : session number, str (session1 (se1) or session2 (se2))

    Output:
    -------
    Return None
    '''
    try:
        if sb == None and se == None:
            data.to_csv(f'{path}\\{name_kS}_{name}.csv')
            with open(f'{pathpckl}\\{name_kS}_{name}.pckl','wb') as pickle_file:
                dump(data, pickle_file)
        else:
            data.to_csv(f'{path}\\{name_kS}_{name}.csv')
            with open(f'{pathpckl}\\{name_kS}_{name}.pckl','wb') as pickle_file:
                dump(data, pickle_file)
        # if
    except:
        strLog = f'\ncsv and pickle file not created!'
        print(strLog)  
    # try
#

def second_lvl_stats(data, feat1, feat2, feat3, feat4):
    '''
    Returns the 2nd level statistics of the data

    Input:
    data: filtered eeg steams (mne object)
    attr: the name of the attributes used
                 (Mean, Std, Kurtosis, Skewness)
    
    Output: 
    -------
    :Return stat of stats
    '''

    stat_stats = {'mean_of_means': None, 'std_of_means': None,
                  'mean_of_stds' : None, 'std_of_stds' : None,
                  'mean_of_kurtosis': None, 'std_of_kurtosis': None,
                  'mean_of_skewness': None, 'std_of_skewness': None}
    
    stat_stats['mean_of_means']    = data[feat1].mean()
    stat_stats['std_of_means']     = data[feat1].std()
    stat_stats['mean_of_stds']     = data[feat2].mean()
    stat_stats['std_of_stds']      = data[feat2].std() 
    stat_stats['mean_of_kurtosis'] = data[feat3].mean()
    stat_stats['std_of_kurtosis']  = data[feat3].std()
    stat_stats['mean_of_skewness'] = data[feat4].mean()
    stat_stats['std_of_skewness']  = data[feat4].std()

    return stat_stats
#


def stats_gnrtr(data, query_output, rsltQuery, bandtouse, numSoi):
    '''
    Generates the statistical features (1st and 2nd lvl stats)
    
    Input:
    -----
    data        : All SoIs (mne object) [num_Soi, len_Soi]
    query_output: Query output (dict)
    bandtouse   : what band is used
    numSoi      : total number of sois
    
    Output:
    -------
    : Returns ss_feature_all: Statistics of windows' statistics [num_Soi*num_channels,num_feats]
    '''
    allFeats          = dict()

    num_rtempDf       = numWdw_gl*num_ch_ui         # num of row per soi in stats
    num_rAlltempDf    = numSoi*num_rtempDf          # num of row for all soi in stats
    num_rAlltempDf_ss = num_rtempDf_ss_gl*numSoi    # num of row for all soi in statstats

    # DataFrames
    Alltemp_DF        = DataFrame(zeros((num_rAlltempDf,len(allTempDF_cols_gl))),columns=allTempDF_cols_gl)
    Alltemp_DF_ss     = DataFrame(zeros((num_rAlltempDf_ss,numCols_ss_ui)),columns=tempDf_ss_cols_gl)
        
    # Storing the channels per soI
    TempDf     = DataFrame.from_dict(zeros((num_rtempDf,numCol_gl)))  
    TempDf_ss  = DataFrame.from_dict(zeros((num_rtempDf_ss_gl,numCols_ss_ui)))

    # Loop for each soi begins
    for i_soi in range(len(data)):
        sm       = rsltQuery['semester'].values[i_soi]
        sb       = rsltQuery['subject'].values[i_soi]
        se       = rsltQuery['session'].values[i_soi]
        keyStmls = rsltQuery['keyStmls'].values[i_soi]
        soi_ID   = rsltQuery['key_soi'].values[i_soi]
        name_kS  = rsltQuery['event_code'].values[i_soi]
        trialID  = rsltQuery['trialID'].values[i_soi]
        
        # Get the sample freq
        sfreq_eeg = data[i_soi][bandtouse].info['sfreq']

        # Put data into dataframe
        data[i_soi] = data[i_soi][bandtouse].copy().to_data_frame(scalings=dict(eeg=1,mag=1,grad=1))
        
        # For each channel in each soi
        for i_ch,ch in enumerate(CfgEEG['eeg_channels_gl']):
            soi_eeg  = data[i_soi][ch]
            soi_eeg  = soi_eeg.to_numpy()
            soi_eeg  = soi_eeg.T

            windowStats      = dict()
            channel_WinStat  = dict()
            
            # Main eeg
            # window_width    = len(soi_eeg)//2
            # num_nonOvrlap   = int(np.ceil(dur_nonOvrlap * sfreq_eeg)) #number of samples not overlapping
            window_width    = int(len(soi_eeg)/num_windws_n0_ui)
            len_nonOvrlap   = int(window_width * per_nonOvrlap_ui)

            # Main eeg
            eeg_feature_wdw = getwindow_stats(window_width, soi_eeg, len_nonOvrlap)
            
            # Saving each window in a dataframe
            windowStats[ch]  = DataFrame.from_dict(eeg_feature_wdw, orient='index',
                                                   columns=colNamesStat_ui).rename_axis('Window-ID')
            
            # Rest the index
            windowStats[ch]  = windowStats[ch].reset_index()
            
            #Statistics of statistics #Mean of means, mean of STDs, std of STDs per channel
            main_eeg_stat_stat = second_lvl_stats(windowStats[ch], mean_srch_gl, std_srch_gl, 
                                                  kurt_srch_gl, skew_srch_gl)

            #Putting Central-Dispersion values into DataFrame
            stat_Stat_df  = DataFrame.from_dict(main_eeg_stat_stat, orient='index',
                                                columns=colNamesStat_ui).rename_axis('stat_type')
            statStats_names = stat_Stat_df.index

            # Manipulating Stat_Stat_df (Puts all the values in a column and remove all the nans by columns)
            statStat_df = stat_Stat_df.melt().dropna().rename(columns={'value':'statstat_value', 'variable':'stat_type'})
            statStat_df['stat_type'] = statStats_names
            
            # Store 1st and 2nd level statistics together
            channel_WinStat[(keyStmls,soi_ID)] = {f'{ch}': windowStats[ch],
                                                  f'StatStats_{ch}': statStat_df}
            
            # Add columns names
            TempDf.columns     = allTempDF_cols_gl
            TempDf_ss.columns  = tempDf_ss_cols_gl 
      
            # Stat
            TempDf['Semester'] = [sm for i in range(0, num_ch_ui) for j in range(0,numWdw_gl)]
            TempDf['Subjects'] = [sb for i in range(0, num_ch_ui) for j in range(0,numWdw_gl)]
            TempDf['Sessions'] = [se for i in range(0, num_ch_ui) for j in range(0,numWdw_gl)]
            TempDf['Channels'] = [CfgEEG['eeg_channels_gl'][i] for i in range(0, num_ch_ui) for j in range(0,numWdw_gl)]
            TempDf['Trial_ID'] = [trialID  for i in range(0,num_ch_ui) for j in range(0,numWdw_gl)]
            TempDf['soI_ID']   = [soi_ID for i in range(0, num_ch_ui) for j in range(0,numWdw_gl)]
        
            # Stat_Stats
            TempDf_ss['Semester'] = [sm for i in range(0, num_ch_ui) for j in range(0,numWdwStats_ui)]
            TempDf_ss['Subjects'] = [sb for i in range(0, num_ch_ui) for j in range(0,numWdwStats_ui)]
            TempDf_ss['Sessions'] = [se for i in range(0, num_ch_ui) for j in range(0,numWdwStats_ui)]
            TempDf_ss['Channels'] = [CfgEEG['eeg_channels_gl'][i] for i in range(0, num_ch_ui) for j in range(0,numWdwStats_ui)]
            TempDf_ss['Trial_ID'] = [trialID  for i in range(0,num_ch_ui) for j in range(0,numWdwStats_ui)]
            TempDf_ss['soI_ID']   = [soi_ID for i in range(0, num_ch_ui) for j in range(0,numWdwStats_ui)]

            
            TempDf.iloc[i_ch*numWdw_gl:numWdw_gl*(i_ch+1),bgn_col_gl:]                = channel_WinStat[(keyStmls,soi_ID)][ch]
            TempDf_ss.iloc[i_ch*(numWdwStats_ui):numWdwStats_ui*(i_ch+1),bgn_col_gl:] = channel_WinStat[(keyStmls,soi_ID)][f'StatStats_{ch}']

            del windowStats; del channel_WinStat
        #for i_ch
        
        sois_ = {'stats':{'Alltemp_strt':i_soi*num_rtempDf, 'Alltemp_end':num_rtempDf*(i_soi+1)},
                 'statstats':{'Alltemp_strt':i_soi*num_rtempDf_ss_gl, 'Alltemp_end':num_rtempDf_ss_gl*(i_soi+1)}} 
        
        # Saving all the trials stats into AllTemp_DF
        Alltemp_DF.iloc[sois_['stats']['Alltemp_strt']:sois_['stats']['Alltemp_end'],:] = TempDf  

        # Saving all the sois statstats into a Alltemp_DF_ss 
        Alltemp_DF_ss.iloc[sois_['statstats']['Alltemp_strt']:sois_['statstats']['Alltemp_end'],:] = TempDf_ss
    # for i_soi

    # Divide the stat-stats df into separate feature df and save
    df_allstatstats = Alltemp_DF_ss
    for i_sm in query_output['semester']:
        stats_sm     = Alltemp_DF.loc[Alltemp_DF['Semester'].isin([i_sm])]
        statstats_sm = Alltemp_DF_ss.loc[Alltemp_DF_ss['Semester'].isin([i_sm])]
        for i_sb in query_output['subject']:
            stats_sb     = stats_sm.loc[stats_sm['Subjects'].isin([i_sb])]
            statstats_sb = statstats_sm.loc[statstats_sm['Subjects'].isin([i_sb])]
            for se in query_output['session']:
                stat_se      = stats_sb.loc[stats_sb['Sessions'].isin([se])]
                statstats_se = statstats_sb.loc[statstats_sb['Sessions'].isin([se])]
                
                # Paths to save the data   
                pathWinStat           = f'{pathRootSoi_ui}{i_sm}\\{i_sb}\\{se}\\stats_csv'; osMakedirs(pathWinStat, exist_ok=True)
                pathWinStatPickle     = f'{pathRootSoi_ui}{i_sm}\\{i_sb}\\{se}\\stats_pckl'; osMakedirs(pathWinStatPickle, exist_ok=True)
                pathWinStatStat       = f'{pathRootSoi_ui}{i_sm}\\{i_sb}\\{se}\\stat_stats_csv'; osMakedirs(pathWinStatStat, exist_ok=True)       
                pathWinStatStatPickle = f'{pathRootSoi_ui}{i_sm}\\{i_sb}\\{se}\\stat_stats_pckl'; osMakedirs(pathWinStatStatPickle, exist_ok=True)    

                # Save the stats data
                save_data(stat_se, pathWinStat, pathWinStatPickle, name_kS+eyeState_ui, i_sb, se, f'stats_{bandtouse}')

                for i_feats in statstats_se['stat_type'].unique():

                    # sort using the soI_ID
                    allFeats[i_feats] = statstats_se.loc[statstats_se['stat_type'].isin([i_feats])].reset_index()   

                    # Drop an unwanted column
                    allFeats[i_feats] = allFeats[i_feats].drop(columns=['index'],axis=1)
                    
                    # save each feature
                    save_data(allFeats[i_feats], pathWinStatStat, pathWinStatStatPickle, name_kS+eyeState_ui, i_sb, se, i_feats+'_'+bandtouse)
                # for
            # for i_se
        # for i_sb
    # for i_sm
    
    del Alltemp_DF; del Alltemp_DF_ss

    # Storing all into one dataframe
    ssFeat_all = DataFrame(zeros((len(df_allstatstats)//numWdwStats_ui,len(ssCol_gl))),columns=ssCol_gl)

    #['Semester','Subjects','Sessions','Channels','Trial_ID','soI_ID']['stat_type'] ['statstat_values']
    #[0,1,2,3,4,5,7]: ['Semester','Subjects','Sessions','Channels','Trial_ID','soI_ID'] ['statstat_values']
    ssFeat_all.iloc[:, [0,1,2,3,4,5,6]] = df_allstatstats.loc[df_allstatstats['stat_type'].isin(['mean_of_means'])].reset_index().iloc[:,[1,2,3,4,5,6,8]]
    ssFeat_all.iloc[:, 7]               = df_allstatstats.loc[df_allstatstats['stat_type'].isin(['std_of_means'])].reset_index().iloc[:,8:]  #"7:" instead of "7" in case we may add more columns
    ssFeat_all.iloc[:, 8]               = df_allstatstats.loc[df_allstatstats['stat_type'].isin(['mean_of_stds'])].reset_index().iloc[:,8:]
    ssFeat_all.iloc[:, 9]               = df_allstatstats.loc[df_allstatstats['stat_type'].isin(['std_of_stds'])].reset_index().iloc[:,8:]
    ssFeat_all.iloc[:, 10]              = df_allstatstats.loc[df_allstatstats['stat_type'].isin(['mean_of_kurtosis'])].reset_index().iloc[:,8:]
    ssFeat_all.iloc[:, 11]              = df_allstatstats.loc[df_allstatstats['stat_type'].isin(['std_of_kurtosis'])].reset_index().iloc[:,8:]
    ssFeat_all.iloc[:, 12]              = df_allstatstats.loc[df_allstatstats['stat_type'].isin(['mean_of_skewness'])].reset_index().iloc[:,8:]
    ssFeat_all.iloc[:, 13]              = df_allstatstats.loc[df_allstatstats['stat_type'].isin(['std_of_skewness'])].reset_index().iloc[:,8:]

    # drop unwanted columns
    ssFeat_all['soI_ID']      = ssFeat_all['soI_ID'].astype(int)
    ssFeat_all['Trial_ID']    = ssFeat_all['Trial_ID'].astype(int)
    ssFeat_all['class_label'] = [cSubjects_ui[ssFeat_all.loc[:,'Subjects'][i]] for i in range(0, len(ssFeat_all))]
    ssFeat_all                = ssFeat_all.loc[:,toKeepCol_gl]
    
    # path to save data
    pathFeat = f'{pathInput_ui}\\{bandtouse}\\FeatureMap'; osMakedirs(pathFeat, exist_ok=True)

    # Save all sois
    save_data(ssFeat_all, pathFeat, pathFeat, name_kS=name_kS, name=f'all_ss_feat')

    return ssFeat_all
#