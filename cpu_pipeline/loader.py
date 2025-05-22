module_name = 'loadPrepare'

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
from config import ConfigPrePrcss as CfgPrePrcss, ConfigMain as CfgMain, ConfigST as CfgST, ConfigStatF as CfgSF, ConfigExtrctSoi as CfgExt, ConfigEEG as CfgEEG
from ui     import LoadPrepare_ui as LdPrep_ui, All_ui 

# OTHER IMPORTS
from astropy.convolution     import RickerWavelet1DKernel as RWK
from copy                    import deepcopy as dpcpy
from mne                     import create_info
from mne.io                  import RawArray
from numpy                   import convolve, zeros, array, empty, nan, unique, asarray, float64, swapaxes, mgrid, isnan, round, min, median, sum, abs, exp, roll
from os                      import makedirs as osMakedirs
from pandas                  import DataFrame, read_pickle, concat, ExcelWriter
from pickle                  import dump, load
from scipy.interpolate       import RBFInterpolator
from sklearn.model_selection import train_test_split
from util                    import projection_XYZtoXY

# USER INTERFACE
# ALL
num_ch_ui        = CfgPrePrcss['num_ch_gl']
pathSoiRoot_ui   = All_ui['pathSoiRoot_ui']
pathInput_ui     = f'{All_ui['pathInput_ui']}'  # f'{All_ui['pathInput_ui']}{bandToUse_ui[0]}'
preRepFnctn_ui   = LdPrep_ui['preRepFnctn']
pathOutput_ui    = f'{All_ui['pathOutput_ui']}' # f'{All_ui['pathOutput_ui']}{bandToUse_ui[0]}'
soiDfAll_file_ui = All_ui['soiDfAll_file_ui']
sgnalCnFnctn_ui  = LdPrep_ui['sgnalConFnctn']

# ExtractSoi
incldMotion_ui   = LdPrep_ui['incldMotion_ui']
num_soi2Use_ui   = LdPrep_ui['num_soi2Use_ui']
querySpecial_ui  = LdPrep_ui['querySpecial_ui']  # for bk_pic, eye state = {EO, EC}
thSkipSoi_ui     = LdPrep_ui['thSkipSoi_ui']
thrshldImp_ui    = LdPrep_ui['thrshldImp_ui']

# PreProcess
applyArtifact_ui = LdPrep_ui['applyArtifact_ui']
applyRef_ui      = LdPrep_ui['applyRef_ui']
applyBandPass_ui = CfgPrePrcss['applyBandPass_gl']
applyNotch_ui    = CfgPrePrcss['applyNotch_gl']
applyImp_ui      = CfgPrePrcss['applyImp_gl']
bandToUse_ui     = LdPrep_ui['bandToUse_ui']
exportPP_ui      = LdPrep_ui['exportPP_ui']
exportPR_ui      = LdPrep_ui['exportPR_ui']
freqsNotch_ui    = CfgPrePrcss['freqsNotch_gl']
highFreq_ui      = CfgPrePrcss['highFreq_gl']
lowFreq_ui       = CfgPrePrcss['lowerFreq_gl']
thrshSkip_ui     = LdPrep_ui['thrshSkip_ui']
useSoftMin_ui    = LdPrep_ui['useSoftMin_ui']
useInv_ui        = LdPrep_ui['useInv_ui']
useLOM_ui        = LdPrep_ui['useLOM_ui']
useLOG_ui        = LdPrep_ui['useLOG_ui']

# SpatialTemporal
cSubjects_ui      = All_ui['cSubjects_ui']
durationWindow_ui = LdPrep_ui['durationWindow_ui']

# CONSTANTS
feature_gl       = CfgSF['feature_gl']
MONTAGE_STD_GL   = CfgEEG['MONTAGE_STD']
NUM_CORDNATE_GL  = CfgExt['NUM_CORDNATE_GL']
RANDOM_STATE_GL  = LdPrep_ui['RANDOM_STATE_GL']
window_gl        = CfgST['window_gl']

# CONFIGURATION
beta_gl          = CfgPrePrcss['betaSftmin_gl']
thrs_gl          = CfgPrePrcss['thrshld_gl']
col_locEEG_gl    = CfgMain['col_locEEG_gl']

# INITIALIZATION
chNameLst_gl        = CfgEEG['eeg_channels_gl'][:num_ch_ui]
inv_data_gl         = {}
leftExtnt_gl        = All_ui['leftExtnt_ui']
lkUpTable_gl        = CfgMain['folder_model_gl']
rghtExtnt_gl        = All_ui['rghtExtnt_ui']
numSoiDf_cols_gl    = CfgExt['numSoiDf_cols_gl']
rawEEG_file_gl      = CfgExt['rawEEG_file_gl']
ricker_1d_kernel_gl = RWK(10)
softmin_data_gl     = {}
test_size_gl        = CfgPrePrcss['test_size_gl']


def extractSoI(stmlusName,colmnGrpBy1st,colmnGrpBy2nd,queryVal,useStdMontage):
    '''
    Input:
    ------
    stiname      : name of stimulus
    colmnGrpBy1st: column name for first  level of grouping
    colmnGrpBy2nd: column name for second level of grouping
    queryVal     : Query values

    Output:
    -------
    returns rawmne sois, rsltQuery, eeg locs and infoSoiPart_all
    '''
    # Paths
    pathEegLocs = f'{pathInput_ui}\\locEEG'; osMakedirs(pathEegLocs, exist_ok=True)
    pathInfo    = f'{pathInput_ui}\\InfoSoI'; osMakedirs(pathInfo, exist_ok=True)

    # Create a df for rsltQuery
    rsltQuery_df = DataFrame(columns=numSoiDf_cols_gl)

    # Get Query Description.
    # region
    # event_code: name of event 
    # semester: list of semester used for the analysis, subject: name of ID subjects
    # session : names of sessions, trialID: unique ID, eyestate: EO (eye open), EC (eye close) 

    with open(pathSoiRoot_ui+soiDfAll_file_ui, 'rb') as fp:
        soiDfAll = read_pickle(fp)
    #

    qStr = ''
    for iq,item in enumerate(queryVal.items()):
        if item[1] != ():
            qStr += f'{item[0]} in {item[1]} and '
        # if
    # for
    qStr = qStr[:-5]

    #Retrieve SoIs' info
    rsltQuery = soiDfAll.query(qStr)
 
    # Pick samples
    if num_soi2Use_ui is not None:
       rsltQuery = rsltQuery.groupby(['semester','subject','session']).sample(n=num_soi2Use_ui,random_state=RANDOM_STATE_GL)
    # if

    # Length of Soi
    rsltQuery['len_Soi'] = rsltQuery['time_end'] - rsltQuery['time_begin']

    rsltQuery = rsltQuery.query(f"{thSkipSoi_ui [0]} < len_Soi and len_Soi < {thSkipSoi_ui [1]}")

    if querySpecial_ui:
        if queryVal['event_code'][0] == 'bk_pic' and queryVal['eye_state'][0] in ['EC','EO']:
            rsltQuery['sti_grp'] = rsltQuery['source_file'].str[:3]
            
            if incldMotion_ui:
                rsltQuery = rsltQuery.query(f"sti_grp in ['1_0','2_0','3_0','2_1','3_1','3_2']") # Baseline+DS+Motion stimuli group
            else:
                rsltQuery = rsltQuery.query(f"sti_grp in ['1_0','2_0','3_0','2_1']") # Baseline+DS stimuli group
            # if
        # if

        rsltQuery = rsltQuery.drop(columns=['sti_grp'])
    # if

    # Drop the unnecessary columns
    rsltQuery = rsltQuery.drop(columns=['len_Soi'])

    pathSoI_lst = rsltQuery['export_path'] + rsltQuery['export_file'] + '.pckl'
    
    # Know the number of soi per session   
    rsltQuery_df['eye_state']  = rsltQuery['eye_state']
    rsltQuery_df['event_Type'] = rsltQuery['event_Type']
    rsltQuery_df['event_code'] = rsltQuery['event_code']
    rsltQuery_df['key_soi']    = rsltQuery['key_soi']
    rsltQuery_df['keyStmls']   = rsltQuery['keyStmls']
    rsltQuery_df['semester']   = rsltQuery['semester']
    rsltQuery_df['subject']    = rsltQuery['subject']
    rsltQuery_df['session']    = rsltQuery['session']
    rsltQuery_df['trialID']    = rsltQuery['trialID']
    
    # rsltQuery_df is a slice of rsltQuery with certain columns
    try:
        with open(f'{pathEegLocs}\\{stmlusName}_rsltQuery.pckl','wb') as pf:
            dump(rsltQuery_df, pf)
    except Exception as e:
        print(f'Error in exporting {stmlusName}_rsltQuery.pckl:\n\t{e}',flush=True)
    # try

    # Find the count of soi and the total soi 
    totalNumSoI = len(rsltQuery_df)

    numSoi_SbSe = rsltQuery_df.groupby(colmnGrpBy1st)[colmnGrpBy2nd].value_counts().reset_index()
    #endregion

    # Extract SoIs
    #region
    begIndx_all          = {}
    endIdx_all           = {}
    i_bgn_tSstmp_soi_all = {}
    i_end_mrk_soi_all    = {} 
    rawMneAll            = dict()
    sfreq_imp_all        = {}
    sfreq_all            = {}
    trialID_all          = {}
    
    # Reading the SoIs
    for i_f,f in enumerate(pathSoI_lst):
        print(f'\n-------  processing {i_f}. file {f} ----------------------\n', flush=True)

        sm	 	    = rsltQuery['semester'].values[i_f]
        sb		    = rsltQuery['subject'].values[i_f]
        se 		    = rsltQuery['session'].values[i_f]
        i_m 		= rsltQuery['marker_ID'].values[i_f]
        keyStml 	= rsltQuery['keyStmls'].values[i_f]
        soi_ID 	    = rsltQuery['key_soi'].values[i_f]
        name_kS 	= rsltQuery['marker_label'].values[i_f]
        trialID 	= rsltQuery['trialID'].values[i_f]
        trial_group = rsltQuery['trial_group'].values[i_f]
        stiType 	= rsltQuery['source_file'].values[i_f][:5]
        begin_indx  = rsltQuery['begin_indx'].values[i_f]
        end_indx 	= rsltQuery['end_indx'].values[i_f]                
            
        infoStrmTpl = (sm,sb,se,i_m,keyStml,soi_ID,name_kS,trialID,trial_group,stiType,begin_indx,end_indx)

        #Load EEG Stream Object
        with open(f"{pathSoiRoot_ui}{sm}\\{sb}\\{se}\\strm\\{rawEEG_file_gl}_{sm}-{sb}-{se}-{stiType}.pckl", 'rb') as fp_soi:
            rawEEG = load(fp_soi)
        # with

        if i_f == 0: chNames = chNameLst_gl

        #Extract SoI from the entire EEG stream
        i_bgn_tSstmp_soi = begin_indx - leftExtnt_gl
        i_end_mrk_soi    = end_indx   + rghtExtnt_gl

        sfreq_eeg = rawEEG['info']['effective_srate']
        sfreq_imp = sfreq_eeg/4 # rawEEG['info']['sfreq_imp']

        if useStdMontage:
            # Standard montage
            infoMne = create_info(ch_names=chNames[:num_ch_ui], ch_types='eeg', 
                                  sfreq = sfreq_eeg).set_montage(MONTAGE_STD_GL, match_case=False)                
        else: #Default: Use actual XYZ-coordinates from the stream
            infoMne  = create_info(ch_names = chNames[:num_ch_ui], ch_types=CfgEEG['TYPE_CH'][:num_ch_ui], sfreq = sfreq_eeg)  
        # if

        #!!!! CAUTION: dpcpy(*) raised memory error. be aware rawMneAll WILL NEVER BE changed!!!!
        rawMneAll[i_f]            = RawArray(rawEEG['series'][:num_ch_ui], infoMne, copy='data', verbose=False)
        sfreq_imp_all[i_f]        = sfreq_imp
        sfreq_all[i_f]            = sfreq_eeg
        i_bgn_tSstmp_soi_all[i_f] = i_bgn_tSstmp_soi
        i_end_mrk_soi_all[i_f]    = i_end_mrk_soi
        begIndx_all[i_f]          = begin_indx
        endIdx_all[i_f]           = end_indx
        trialID_all[i_f]          = trialID
    # for i_f

    infoSoiPart_all = {'sfreqimp':sfreq_imp_all,'bgntSstmp':i_bgn_tSstmp_soi_all,'endmrk':i_end_mrk_soi_all,
                       'beginidx':begIndx_all,'endidx': endIdx_all,'trialID':trialID_all,'sfreq':sfreq_all}

    # Electrode locations
    if useStdMontage:
       name   = 'useStdMontage'
       loc_ch = rawMneAll[0]._get_channel_positions()
    else:
       name   = 'noUseStdMontage'
       loc_ch = array([list(rawEEG['info']['channels'][i]['location'][0].values()) for i in range(0, num_ch_ui)], dtype=float)
       loc_ch = loc_ch.reshape(num_ch_ui, NUM_CORDNATE_GL)
    # if

    try:
        with open(f'{pathEegLocs}\\loc_channel_{name}.pckl','wb') as fp:
            dump(loc_ch, fp)
    except Exception as e:
        print(f'Error in exporting {pathEegLocs}\\loc_channel.pckl:\n\t{e}',flush=True)
    # with

    try:
        loc_eeg = DataFrame(loc_ch, index=CfgEEG['eeg_channels_gl'], columns=col_locEEG_gl)
        with ExcelWriter(f'{pathEegLocs}\\loc_channel_{name}.xlsx') as loc_fp:
            loc_eeg.to_excel(loc_fp, sheet_name='EEG_locs')
    except Exception as e:
        print(f'Error in exporting {pathEegLocs}/loc_channel.xlsx:\n\t{e}',flush=True)
    #
    
    try:
        with open(f'{pathInfo}\\infoSoiPartAll.pckl','wb') as pf:  
            dump(infoSoiPart_all, pf)
    except Exception as e:
        print(f'Error in exporting {pathInfo}/infoSoiPartAll.pckl:\n\t{e}',flush=True)
    #
    print('Load data function ended...', flush=True)

    #endregion
    return rawMneAll, infoSoiPart_all, rsltQuery_df, loc_ch, totalNumSoI
#

def get_windows(window_width, data, len_nonOvrlapAdjstd, num_WdwsAdjstd):
    '''
    Slides using a window to get [chs, timesample] data
    or any data

    Input:
    -----
    window_width : the length of the window in number of samples
    data         : input data (1D vector) [len_Soi]
    len_nonOvrlapAdjstd: the overlap duration

    Output:
    -------  
    Return: spatiotemporal data
    '''
    timeWdw    = empty((num_ch_ui,window_width,num_WdwsAdjstd))
    timeWdw[:] = nan

    data = data.to_numpy()

    for i_wndw in range(0, num_WdwsAdjstd):
        i_bgn = i_wndw*len_nonOvrlapAdjstd
        data_windw = data[:,i_bgn:(i_bgn + window_width)]

        if data_windw.shape[1] < window_width:
            break
        # if

        timeWdw[:,:,i_wndw] = data_windw
    # for
    return timeWdw
#

def preProcessAfterBand(streamAll,infoSoiPart_all,bandToUse,rsltQuery):
    '''
    Apply PreProcessing to entire stream, Conditioning & Pre-Representation after extracting the Band

    Input:
    -----
    streamAll      : raw EEG signals (mne object) [num_Soi, len_Soi]
    infoSoiPart_all: contains informations of each raw EEG signal
    bandToUse      : what band is used
    rsltQury       : dataframe with all infos for each soi

    Output:
    -------
    : Returns PP or PR signals [num_Soi, len_Soi]
    '''

    # Path
    pathNotGoodSoi = f'{pathInput_ui}\\NotGoodSois'; osMakedirs(pathNotGoodSoi, exist_ok=True)
    pathPP         = f'{pathInput_ui}\\PP'; osMakedirs(pathPP, exist_ok=True)

    soiMneBand     = {bandToUse: None}
    notGood        = {}
    soiAllFtrBand  = {}
    soiFltrdBand   = {}
    soiAllCndtn    = {}

    # Create data_info for using the rsltQuery
    dataset_info                 = rsltQuery.loc[:,['semester','subject','session']].reset_index().drop(columns=['index'])
    dataset_info['class_labels'] = [cSubjects_ui[dataset_info.loc[:,'subject'][i]] for i in range(0, len(dataset_info))]
    
    for i_soi, rawEEGmne in streamAll.items():  
        if infoSoiPart_all['endmrk'][i_soi] - rawEEGmne.n_times > thrshSkip_ui:
            notGood[i_soi] = i_soi
            
            # Logging 
            strLog = (f'{i_soi} end_indx {infoSoiPart_all['endmrk'][i_soi]}is greater than the length {rawEEGmne.n_times} of the entire stream.' 
                      f'So len(SoI) {infoSoiPart_all['endmrk'][i_soi]-infoSoiPart_all['bgntSstmp'][i_soi]}is shorter than expected! Not included to dataset')
            with open(f'{pathNotGoodSoi}\\catchErrorLog.txt','w') as fErrLog:
                fErrLog.write(strLog)

            continue
        # if

        #test
        # if i_soi == 395:
        #     print()

        infoMne = rawEEGmne.info
        
        # PREPROCESSING
        if applyArtifact_ui:
            ...
        #
        if applyRef_ui:
            #mne.io._apply_reference
            ...
        #
        if applyNotch_ui:
            EEGmne_fltrd = rawEEGmne.copy().notch_filter(freqsNotch_ui, verbose=False) 
        #
        if applyImp_ui:
            EEGmne_fltrd.filter(l_freq=infoSoiPart_all['sfreqimp'][i_soi] + lowFreq_ui, h_freq=infoSoiPart_all['sfreqimp'][i_soi] - lowFreq_ui, 
                                method='iir',verbose=False)
        if applyBandPass_ui:
            EEGmne_fltrd.filter(l_freq=lowFreq_ui, h_freq=highFreq_ui, method='iir',verbose=False)
        #

        # EXTRACT BAND
        if bandToUse not in bandToUse_ui:
            return f'{bandToUse} is not in the list of bands'
        else:
            EEGmne_fltrdBand = EEGmne_fltrd.copy().filter(l_freq=CfgEEG['bandEEG_gl'][bandToUse][0], h_freq=CfgEEG['bandEEG_gl'][bandToUse][1], 
                                                          method='iir', verbose = False) 
        # if

        del EEGmne_fltrd 
        
        # Get the SoIs.
        soiFltrdBand[i_soi] = EEGmne_fltrdBand.copy().get_data(start=infoSoiPart_all['bgntSstmp'][i_soi], stop=infoSoiPart_all['endmrk'][i_soi])
        
        # SIGNAL CONDITIONING
        if useLOM_ui:
            fNamePP = 'sLom'
            soiAllCndtn[i_soi] = RawArray(norm_LoM(soiFltrdBand[i_soi]), infoMne, copy='data') 
        elif useLOG_ui:
            fNamePP = 'LoG'
            soiAllCndtn[i_soi] = RawArray(norm_LoG(soiFltrdBand[i_soi]), infoMne, copy='data')  
        else:
            fNamePP = 'nan'
            soiAllCndtn[i_soi] = RawArray(soiFltrdBand[i_soi], infoMne, copy='data')
        # if

        soiMneBand[bandToUse] = dpcpy(soiAllCndtn[i_soi])  
        soiAllFtrBand[i_soi]  = dpcpy(soiMneBand)
    # for i_soi
    
    # PRE-REPRESENTATION
    if useSoftMin_ui: # Inverse_ Representation
        fNamePP = f'{fNamePP}_{preRepFnctn_ui}'
        soiAllPR = Softmin(soiAllFtrBand, rsltQuery)
    elif useInv_ui:   # Inverse Representation
        fNamePP = f'{fNamePP}_{sgnalCnFnctn_ui}'
        soiAllPR = InverseWeight(soiAllFtrBand, rsltQuery)
    else:
        soiAllPR = soiAllFtrBand
    # if

    with open(f'{pathNotGoodSoi}\\notGood_soi.pckl','wb') as fp:
        dump(notGood, fp)
    # with

    # Export the PP (PreProcessed) & PPEh (PreProcessed + Enhanced)
    if exportPP_ui:
        try:
            with open(f'{pathPP}\\soiAllFtrBand.pckl','wb') as pf:  
                dump(soiAllFtrBand, pf)
        except Exception as e:
            print(f'Error in exporting {pathPP}/soiAllFtrBand.pckl:\n\t{e}',flush=True)
        # try
    # if

    if exportPR_ui:
        try:
            with open(f'{pathPP}\\soiAllPR.pckl','wb') as pf:  
                dump(soiAllPR, pf)
        except Exception as e:
            print(f'Error in exporting {pathPP}/soiAllPR.pckl:\n\t{e}',flush=True)
        # try
    # if
    return dataset_info, soiFltrdBand, soiAllCndtn, soiAllPR 
#

def preProcessBeforeBand(streamAll,infoSoiPart_all,bandToUse,rsltQuery):
    '''
    Apply PreProcessing to entire stream, Conditioning & Pre-Representation before extracting the Band

    Input:
    -----
    streamAll      : raw EEG signals (mne object) [num_Soi, len_Soi]
    infoSoiPart_all: contains informations of each raw EEG signal
    bandToUse      : what band is used
    rsltQury       : dataframe with all infos for each soi

    Output:
    -------
    : Returns PP or PR signals [num_Soi, len_Soi]
    '''

    # Path
    pathNotGoodSoi   = f'{pathInput_ui}\\NotGoodSois'; osMakedirs(pathNotGoodSoi, exist_ok=True)
    pathPP           = f'{pathInput_ui}\\PP'; osMakedirs(pathPP, exist_ok=True)
    soiMneCndtn      = {'None': None}
    soiAllCndtn_temp = {}
    notGood          = {}
    soiAllFltrBand   = {}
    soiAllCndtn      = {}
    soiAllFltrd      = {}

    # Create data_info for using the rsltQuery
    dataset_info                 = rsltQuery.loc[:,['semester','subject','session']].reset_index().drop(columns=['index'])
    dataset_info['class_labels'] = [cSubjects_ui[dataset_info.loc[:,'subject'][i]] for i in range(0, len(dataset_info))]

    for i_soi, rawEEGmne in streamAll.items(): 
        if infoSoiPart_all['endmrk'][i_soi] - rawEEGmne.n_times > thrshSkip_ui:
            notGood[i_soi] = i_soi
        
            # Logging 
            strLog = (f'{i_soi} end_indx {infoSoiPart_all['endmrk'][i_soi]}is greater than the length {rawEEGmne.n_times} of the entire stream.' 
                      f'So len(SoI) {infoSoiPart_all['endmrk'][i_soi]-infoSoiPart_all['bgntSstmp'][i_soi]}is shorter than expected! Not included to dataset')
            with open(f'{pathNotGoodSoi}\\catchErrorLog.txt','w') as fErrLog:
                fErrLog.write(strLog)

            continue
        # if

        #test
        # if i_soi == 395:
        #     print()

        infoMne = rawEEGmne.info

        # PREPROCESSING
        if applyArtifact_ui:
            ...
        #
        if applyRef_ui:
            #mne.io._apply_reference
            ...
        #
        if applyNotch_ui:
            EEGmne_fltrd = rawEEGmne.copy().notch_filter(freqsNotch_ui, verbose=False)
        #
        if applyImp_ui:
            EEGmne_fltrd.filter(l_freq=infoSoiPart_all['sfreqimp'][i_soi] + lowFreq_ui, h_freq=infoSoiPart_all['sfreqimp'][i_soi] - lowFreq_ui, 
                                method='iir',verbose=False) 
        if applyBandPass_ui:
            EEGmne_fltrd.filter(l_freq=lowFreq_ui, h_freq=highFreq_ui, method='iir',verbose=False)
        # if

        # EXTRACT SOI
        soiAllFltrd[i_soi] = EEGmne_fltrd.copy().get_data(start=infoSoiPart_all['bgntSstmp'][i_soi], stop=infoSoiPart_all['endmrk'][i_soi])

        del EEGmne_fltrd

        # SIGNAL CONDITIONING
        if useLOM_ui:
            fNamePP = 'sLom'
            soiAllCndtn[i_soi] = RawArray(norm_LoM(soiAllFltrd[i_soi].copy()), infoMne, copy='data') 
        elif useLOG_ui:
            fNamePP = 'LoG'
            soiAllCndtn[i_soi] = RawArray(norm_LoG(soiAllFltrd[i_soi].copy()), infoMne, copy='data')  
        else:
            fNamePP = 'Nan'
            soiAllCndtn[i_soi] = RawArray(soiAllFltrd[i_soi].copy(), infoMne, copy='data')
        # if
        soiMneCndtn['None'] = dpcpy(soiAllCndtn[i_soi])  # 'None' is for “signal w/o band separation”. It is needed due to PRE-REPRESENTATION functions’ data structure.
        soiAllCndtn_temp[i_soi]  = dpcpy(soiMneCndtn)
    # for

    # PRE-REPRESENTATION
    if useSoftMin_ui: # Inverse_ Representation
        fNamePP = f'{fNamePP}_{preRepFnctn_ui}'
        soiAllPR = Softmin(soiAllCndtn_temp, rsltQuery)
    elif useInv_ui:   # Inverse Representation
        fNamePP = f'{fNamePP}_{sgnalCnFnctn_ui}'
        soiAllPR = InverseWeight(soiAllCndtn_temp, rsltQuery)
    else:
        soiAllPR = soiAllCndtn_temp
    #

    # EXTRACT BAND
    if bandToUse not in bandToUse_ui:
        return f'{bandToUse} is not in the list of bands'
    else:
        for i_soi in soiAllPR.keys():
            soiAllFltrBand[i_soi] = {bandToUse: soiAllPR[i_soi]['None'].copy().filter(l_freq=CfgEEG['bandEEG_gl'][bandToUse][0], h_freq=CfgEEG['bandEEG_gl'][bandToUse][1], 
                                                                                      method='iir', verbose = False)}
        # for
    # if

    with open(f'{pathNotGoodSoi}\\notGood_soi.pckl','wb') as fp:
        dump(notGood, fp)
    # with

    # Export the PP (PreProcessed) & PPEh (PreProcessed + Enhanced)
    if exportPP_ui:
        try:
            with open(f'{pathPP}\\soiAllPR.pckl','wb') as pf:  
                dump(soiAllPR, pf)
        except Exception as e:
            print(f'Error in exporting {pathPP}/soiAllPR.pckl:\n\t{e}',flush=True)
        # try
    # if

    if exportPR_ui:
        try:
            with open(f'{pathPP}\\soiAllFltrBand.pckl','wb') as pf:  
                dump(soiAllFltrBand, pf)
        except Exception as e:
            print(f'Error in exporting {pathPP}/soiAllFltrBand.pckl:\n\t{e}',flush=True)
        # try
    # if
    return dataset_info, soiAllFltrd, soiAllCndtn, soiAllFltrBand
#

def norm_LoG(streamAll):
    """
    Uses norm LOG 
    """
    St_conv  = zeros(streamAll.shape)
    for i in range(streamAll.shape[0]):
        St_conv[i] = convolve(streamAll[i], ricker_1d_kernel_gl, mode='same')
    # for
    return St_conv
#

def norm_LoM(streamAll):
    """
    Takes the raw EEG signals after filtering and uses Laplacian of Mean 
    as a signal conditioning.

    Input:
    -----
    streamAll: filtered raw EEG streamAlls

    Output:
    ------
    : Returns raw EEG signal
    """
    LOM_data = array(zeros(shape=(streamAll.shape[0],streamAll.shape[1]-4)))
    for i_ch in range(len(streamAll)): 
        St             = streamAll[i_ch]
        St_minus_1     = roll(St, 1)   # Shift array St by 1 to get St-1
        St_plus_1      = roll(St, -1)  # Shift array St by -1 to get St+1
        F_avr          = (St_minus_1[1:-1]  + St[1:-1] + St_plus_1[1:-1])  / 3
        F_avr_minus_1  = roll(F_avr, 1)   # Shift array St by 1 to get St-1
        F_avr_plus_1   = roll(F_avr, -1)
        F_avrD_t       = F_avr_plus_1[1:-1] - 2* F_avr[1:-1]  + F_avr_minus_1[1:-1]

        # Store the streamAll
        LOM_data[i_ch,:] = F_avrD_t
    #
    return(LOM_data)
#

def InverseWeight(streamAll, rsltQuery):  
    """
    Computes the inverse weight for each soi

    Input:
    -----
    streamAll : raw mne (entire)
    stmlusName: name for stimulus
    bandToUse : what band is used

    Output:
    -------
    : Returned mne object for inverse weight
    """
    
    # Index list for the rsltQuery
    IndexList = rsltQuery.index 

    Sbj_ID    = unique(rsltQuery["subject"])
    Sessions  = unique(rsltQuery["session"])

    for Sbj in Sbj_ID:
        for Se in Sessions:
            Sbj_Session  = {}
            Sbj_Se       = DataFrame()
            SoI_Id = rsltQuery[(rsltQuery['subject'] == Sbj) & (rsltQuery["session"] == Se)].index
    
            for Chn in CfgEEG['eeg_channels_gl']:
                Chn_Index = CfgEEG['eeg_channels_gl'].index(Chn) # chName_ui[chName_ui == Chn].index[0]
                Chns_Data = DataFrame()

                for SoI in SoI_Id:
                    SoI_Index = IndexList.get_loc(SoI)
                    SoI_Chn   = DataFrame(streamAll[SoI_Index]['None'].get_data()[Chn_Index]) 
                    Chns_Data = concat([Chns_Data , SoI_Chn], axis = 1)
                #SoI
                Chns_Data_axis1 = Chns_Data.interpolate(method='linear', axis = 1)
                Chns_Data_axis0 = Chns_Data_axis1.interpolate(method='linear', axis = 0)
                
                Chns_Data_arr            = array(Chns_Data_axis0.T)
                Ref_Signal               = median(Chns_Data_arr, axis = 0)
                Abs_Diff                 = abs( Chns_Data_arr - Ref_Signal)
                Abs_Diff[Abs_Diff == 0 ] = thrs_gl
                Inv_Abs_Diff             = 1 / Abs_Diff
                Sum_Inv_Abs_Diff         = sum(Inv_Abs_Diff, axis=0)
                Wght_Abs                 = Inv_Abs_Diff / Sum_Inv_Abs_Diff
                Inv_Weighted             = Wght_Abs * Chns_Data_arr

                Sbj_Se = concat([Sbj_Se, DataFrame(Inv_Weighted)] , axis= 0)
            #chn
            
            for SoI in enumerate(SoI_Id):
                SoI_Index = IndexList.get_loc(SoI[1])         
                SoI_Wghtd = zeros((24, Sbj_Se.shape[1]))
                soiBand = {}
                for ind_chn in range(len(CfgEEG['eeg_channels_gl'])):
                    SoI_Wghtd[ind_chn] = Sbj_Se.iloc[SoI[0] + ind_chn * len(SoI_Id)]
                #ind_chn
                n_times = streamAll[SoI_Index]['None'].n_times 

                # Get the info mne
                infoMne = streamAll[SoI_Index]['None'].info
                soiBand['None'] = RawArray(SoI_Wghtd[:,:n_times], infoMne)  
                Sbj_Session[SoI_Index] = soiBand
            #SoI
            inv_data_gl.update(Sbj_Session)
        # for
    # for
    return inv_data_gl
#

def Softmin(streamAll, rsltQuery):  
    '''
    Computes the softmin for each soi

    Input:
    -----
    streamAll : raw mne (any band) [num_Soi, len_Soi]
    stmlusName: name for stimulus
    bandToUse : what band is used

    Output:
    -------
    : Returned mne object for inverse weight [num_Soi, len_Soi]
    '''

    # Index list for the rsltQuery
    IndexList = rsltQuery.index 

    Sbj_ID    = unique(rsltQuery["subject"])
    Sessions  = unique(rsltQuery["session"])

    for Sbj in Sbj_ID:
        for Se in Sessions:
            Sbj_Session  = {}
            Sbj_Se       = DataFrame()
            SoI_Id = rsltQuery[(rsltQuery['subject'] == Sbj) & (rsltQuery["session"] == Se)].index
    
            for Chn in CfgEEG['eeg_channels_gl']:
                Chn_Index = CfgEEG['eeg_channels_gl'].index(Chn) # chName_ui[chName_ui == Chn].index[0]
                Chns_Data = DataFrame()

                for SoI in SoI_Id:
                    SoI_Index = IndexList.get_loc(SoI)
                    SoI_Chn   = DataFrame(streamAll[SoI_Index]['None'].get_data()[Chn_Index]) 
                    Chns_Data = concat([Chns_Data , SoI_Chn], axis = 1)
                #SoI
                Chns_Data_axis1= Chns_Data.interpolate(method='linear', axis = 1)
                Chns_Data_axis0= Chns_Data_axis1.interpolate(method='linear', axis = 0)
                Chns_Data_arr  = array(Chns_Data_axis0.T)
                Ref_Signal     = median(Chns_Data_arr, axis = 0)
                Abs_Diff       = abs(Chns_Data_arr - Ref_Signal)
                Exp_Abs        = exp(Abs_Diff * beta_gl) 
                Sum_Abs_Beta   = sum(Exp_Abs, axis=0)
                Wght_Abs       = Exp_Abs / Sum_Abs_Beta
                Exp_Weighted   = Wght_Abs * Chns_Data_arr
                Sbj_Se= concat([Sbj_Se, DataFrame(Exp_Weighted)] , axis= 0)
            #chn
            
            for SoI in enumerate(SoI_Id):
                SoI_Index = IndexList.get_loc(SoI[1])         
                SoI_Wghtd = zeros((24, Sbj_Se.shape[1]))
                soiBand = {}
                for ind_chn in range(len(CfgEEG['eeg_channels_gl'])):
                    SoI_Wghtd[ind_chn] = Sbj_Se.iloc[SoI[0] + ind_chn * len(SoI_Id)]
                #ind_chn
                n_times = streamAll[SoI_Index]['None'].n_times 

                # Get the info mne
                infoMne = streamAll[SoI_Index]['None'].info
                soiBand['None'] = RawArray(SoI_Wghtd[:,:n_times], infoMne)
                Sbj_Session[SoI_Index] = soiBand
            #SoI
            softmin_data_gl.update(Sbj_Session)
        # for
    # for
    return softmin_data_gl
#

def preProcess_1(applyRef,applyNotch,applyArtifact):     
    if applyRef:
        #mne.io._apply_reference
        ...
    #
    if applyNotch:
        ...
    #
    if applyArtifact:
        ...
    #
#

def spatialTemp_map(data, viewID, bandID):
    '''
    Transforms from 2D to 3D.

    Input:
    -----
    data  : PPEh or PP data mne object with (num_Soi, len_soi)
    viewID: Using 1 to get other views

    Output:
    -------
    : Return interpolated 3D map baseViewTensor (num_Soi,num_channels,num_windows,len_Soi)
    '''


    # Get the minimum SoI duration
    time_minDuration = min(array([data[i][bandID].n_times for i in range(len(data))]))

    # Get the window width associated with the time duration
    window_width = window_gl['window_width'][durationWindow_ui]

    # Get the overlap adjusted value
    len_nonOvrlapAdjstd = int(window_width * window_gl['ratio_Ovrlap'][durationWindow_ui])

    # Number of windows
    num_WdwsAdjstd =  window_gl['k_adjstd_gl'][durationWindow_ui]

    # PPEh = (num_Soi, num_channels, len_Soi, num_windows)
    baseViewTensor    = empty((len(data), num_ch_ui, window_width, num_WdwsAdjstd))
    baseViewTensor[:] = nan

    # Loop over sois
    for i_soi in range(len(data)):
        # Get the soi data
        data[i_soi] = data[i_soi][bandID].to_data_frame(scalings=dict(eeg=1,mag=1,grad=1)).iloc[:time_minDuration,1:num_ch_ui+1]

        # Reshape the soi from (len_Soi,num_channels)-> (num_channels, len_Soi)
        data[i_soi] = data[i_soi].T

        baseViewTensor[i_soi,:,:,:] = get_windows(window_width, data[i_soi], len_nonOvrlapAdjstd, num_WdwsAdjstd)
    # for

    # Remove any nans left
    baseViewTensor = baseViewTensor[~isnan(baseViewTensor).any(axis=(-1,-2))]

    baseViewTensor = baseViewTensor.reshape(len(data),num_ch_ui,window_width,num_WdwsAdjstd)

    if viewID == '2':
        baseViewTensor = swapaxes(data, 2, 3)  # [num_Soi,num_channels,num_windows,len_Soi]
    elif viewID == '3':
        baseViewTensor = swapaxes(data, 1, 2)  # [num_Soi,num_windows,num_channels,len_Soi]
        baseViewTensor = swapaxes(data, 2, 3)  # [num_Soi,num_windows,len_Soi,num_channels]
    #
    return baseViewTensor
#

def pullTrainTest(bandToUse, data, dataset_info, rsltQuery, main_str, trn_str, 
                  tst_str, seed_num, viewID=None):
    '''
    Splits the data into Training, Validation and Testing sets and shuffles.
    (Se1) and (se2)

    Input:
    ------
    viewID       : view numbers (1,2 or 3)
    bandToUse    : what band is used
    data         : Training and testing data (num_Soi, num_features, num_channels)
    dataset_info : dataframe that include all column names of dataset_info
    rsltQuery    : dataframe that include some column names of soiDfAll
    main_str     : column name to pull training and testing data. Eg sessions
    trn_str      : column value to pull training data. Eg se1
    tst_str      : column value to pull testing data. Eg se2

    Output:
    ------
    :Return train (num_Soi, num_feats, num_channels) validation & testing sets(df or numpy)
    '''

    # New indexing
    rsltQueryLkUp = rsltQuery.reset_index()
    dataset_info  = dataset_info.reset_index()

    # Read subjct-session info (session-type and classlabels)
    trn_cLabel = dataset_info.loc[dataset_info[main_str].isin(trn_str)].loc[:,'class_labels']
    trnSess    = dataset_info.loc[dataset_info[main_str].isin(trn_str)].loc[:,main_str]
    tst_cLabel = dataset_info.loc[dataset_info[main_str].isin(tst_str)].loc[:,'class_labels']
    tstSess    = dataset_info.loc[dataset_info[main_str].isin(tst_str)].loc[:,main_str]

    # keep track of the indices of the Training and Testing
    idxTrn = trn_cLabel.index.tolist()
    idxTst = tst_cLabel.index.tolist()

    # Split into Training and Testing
    fTrain, fTest = data[idxTrn,:], data[idxTst,:]

    cTrain, cTest = trn_cLabel[idxTrn], tst_cLabel[idxTst]

    XTrn_sess, XTst_sess = trnSess[idxTrn], tstSess[idxTst]

    # Split the training into training and validation
    fTrn, fVal, cTrn, cVal, fIndices, cIndices = train_test_split(fTrain, cTrain, idxTrn, test_size=test_size_gl, random_state=seed_num, 
                                                                  shuffle=True, stratify=None)

    # Adding a column for Training, Validation and Testing
    rsltQueryLkUp.loc[fIndices, 'ID_names'] = 'Train'
    rsltQueryLkUp.loc[cIndices, 'ID_names'] = 'Val'
    rsltQueryLkUp.loc[idxTst, 'ID_names']   = 'Test'

    # Query for TrainingValidation and Testing for Single Fold
    rsltQueryLkUpTrnVal = rsltQueryLkUp.query(f"{main_str} == {trn_str}")
    rsltQueryLkUpTst    = rsltQueryLkUp.query(f"{main_str} == {tst_str}")

    # Query for Training sets only from the main Query
    rsltQueryTrn = rsltQuery.reset_index().query(f"{main_str} == {trn_str}")

    # Export LookUpTable for Single Fold: TrainVal and Test
    try:
        if viewID != None:
            pathLkTable = f'{pathOutput_ui}\\View{viewID}\\{bandToUse}\\{lkUpTable_gl}'; osMakedirs(pathLkTable, exist_ok=True)
        else:
            pathLkTable = f'{pathOutput_ui}\\{bandToUse}\\{lkUpTable_gl}'; osMakedirs(pathLkTable, exist_ok=True)
        # 

        with (open(f'{pathLkTable}\\TrainVal_LkUp_all_SF.pckl','wb') as pf1, 
              open(f'{pathLkTable}\\Tst_LkUp_all_SF.pckl','wb') as pf2): 
            dump(rsltQueryLkUpTrnVal, pf1)
            dump(rsltQueryLkUpTst,pf2)
        # with
    except Exception as e:
        print(f'Error in exporting pickle file for LookUp table:\n\t{e}',flush=True)
    #
    
    return [(fTrn, cTrn), (fVal, cVal), (fTest, cTest), (rsltQueryTrn,fTrain,cTrain)]
#

def transform_to_TopoMap(df_data, sizeX, sizeY, numSmples, eeg_locs_all):
    '''
    Interpolated the electrodes to the grid and 
    transforms the grid to a map of 2D

    Input:
    -----
    df_data    : statistical features ie [num_samples, num_feats, num_channels] 
    n_gridpts_x: # of gridpoints in x
    n_gridpts_y: # of gridpoints in y
    num_samples: the number of sois 
    normalize  : Boolean

    Output:
    -------
    : Return normalized and interpolated 2D map [num_samples, ngrdipts, ngridpts, num_feats]
    ''' 

    # convert format
    eegLocs24 = asarray(eeg_locs_all[:num_ch_ui], dtype=float64) 

    # locations in 2D
    eegLocs = array([projection_XYZtoXY(eegLocs24[i]) for i in range(len(eegLocs24))])

    # Number of features
    numFeats = df_data.shape[1]

    # Create grid
    gridXY = mgrid[min(eegLocs[:,0]):max(eegLocs[:,0]):sizeX*1j,
                   min(eegLocs[:,1]):max(eegLocs[:,1]):sizeY*1j] 

    # Change shape
    grid_xy = gridXY.reshape(2, -1).T 

    # create the interpolated data
    data_interpltd = zeros([numFeats, numSmples, sizeX, sizeY]) 
    
    for i_soi in range(len(df_data)):
        for idx_feat in range(numFeats):    
            
            #Feature: [num_Soi, num_channels] per feat 
            stat_feats = df_data[i_soi,idx_feat]     
            
            # interpolates the data
            interpolated = RBFInterpolator(eegLocs, stat_feats, kernel='cubic')(grid_xy)
            data_interpltd[idx_feat,i_soi,:,:] = interpolated.reshape((sizeX,sizeY))
        # for i_feats
    # for i_soi
            
    # swap axes
    data_interpltd = swapaxes(asarray(data_interpltd),0,1)  # [num_Soi,num_feats,num_gripts,num_gridpts]
    data_interpltd = swapaxes(data_interpltd, 1, 2)
    data_interpltd = swapaxes(data_interpltd, 2, 3)         # [num_Soi,num_gridpts,num_gridpts,num_feats]
    
    return data_interpltd
#









    
