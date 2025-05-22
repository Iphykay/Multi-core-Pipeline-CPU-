module_name = 'compile_bestscore'

'''
Version: v1.3.1

Description:
   Compiles best scores

Authors:
    Dr. Omer Muhammet Soysal
    Iphy Kelvin

Date Created     : 10/15/2024
Date Last Updated: 10/15/2024

Doc:

Notes:

ToDO:
    Rename 'single'-> 'IPM'; 'overall'-> 'OPM'

'''


# CUSTOM IMPORTS
from config_compile import ConfigMainCmp, ConfigPMCmp
from ui             import All_ui


# OTHER IMPORTS
from csv    import writer as csvW
from numpy  import full, nan
from os     import makedirs as osMakedirs
from pandas import read_excel, DataFrame, read_pickle
from pickle import dump

# USER INTERFACE
combntn_ui    = All_ui['combntn_ui']
pathOutput_ui = All_ui['pathOutput_ui']


# CONSTANTS
dType_sngl_gl     = ConfigMainCmp['dType_sngl_gl'] 
dType_ovr_gl      = ConfigMainCmp['dType_ovr_gl']
numSbjCmb_gl      = ConfigMainCmp['numSbjsAll_ui']
numModels_gl      = ConfigMainCmp['numModels_gl']
pm_gl             = ConfigPMCmp['pmName_gl']
colName_single_gl = ConfigMainCmp['colName_single_gl']
colName_overal_gl = ConfigMainCmp['colName_overal_gl']



def compile_bestscore():
    '''
    Puts the individual performance and overall
    model performance into a dataframe for StatsRepresentation

    Outputs:
    --------
    Returns df individual and df overall
    '''
    lenDf_single  = combntn_ui * numModels_gl * len(pm_gl['IPM']) * numSbjCmb_gl
    lenDf_overal  = combntn_ui * numModels_gl * len(pm_gl['OPM']) 

    # PATHS
    pathgrpLkUp = f'{pathOutput_ui}\\SavingGroups'

    # Read combination group
    grpsLkUp = read_excel(f'{pathgrpLkUp}\\groupLkUp.xlsx').drop('Unnamed: 0', axis=1)
    lstGrps  = list(grpsLkUp.Subjects)

    scores_single  = DataFrame(full((lenDf_single,len(colName_single_gl)), nan),columns=colName_single_gl).astype(dType_sngl_gl)
    scores_overall = DataFrame(full((lenDf_overal,len(colName_overal_gl)), nan),columns=colName_overal_gl).astype(dType_ovr_gl)
    rw_sng = 0
    rw_all = 0
    errLogCntr = -1
    errorLog   = dict()
    for i_grp, grpsbjs in zip(range(combntn_ui),lstGrps):
        pathGrp = f'G{i_grp}'

        for i_dic, (mdTyp, mdLst) in enumerate(numModels_gl.items()):
            for mdl in mdLst:
                pathData = f'{pathOutput_ui}\\{pathGrp}\\entire\\Predict\\Perf\\Best\\{mdl}'

                try:
                    with open(f'{pathData}\\Bestscores.pckl','rb') as fp1:
                        data = read_pickle(fp1)
                    # with
                    for i_sb, SbID in enumerate(eval(grpsbjs)):
                        for i_PM, name_IPM in enumerate(pm_gl['IPM']):
                            scores_single.loc[rw_sng + i_PM,colName_single_gl]   = [mdTyp,mdl,SbID,pathGrp,name_IPM,data['single'][name_IPM][i_sb]]
                        # for
                        rw_sng += i_PM + 1
                    # for
                except Exception as e:
                    errLogCntr = errLogCntr + 1                
                    errorLog[errLogCntr] = f'Single,{pathGrp},{mdl},{e}]'
                # try

                try:
                    # Overall scores
                    for i_PM, name_OPM in enumerate(pm_gl['OPM']):
                        scores_overall.loc[rw_all + i_PM,colName_overal_gl]  = [mdTyp,mdl,pathGrp,name_OPM,data['overal'][name_OPM]]
                    #
                    rw_all += i_PM + 1
                except Exception as e:
                    errLogCntr = errLogCntr + 1                
                    errorLog[errLogCntr] = f'Overall,{pathGrp},{mdl},{e}]'
                # try
            # for mdl
        # for i_dic
    # for i_grp

    return scores_single, scores_overall, errorLog
#

def compile_Fusionbestscore(foldername=str):
    '''
    Puts the individual performance and overall
    model performance into a dataframe for ST-Representation

    Outputs:
    --------
    Returns df individual and df overall
    '''
    lenDf_single  = combntn_ui * numModels_gl * len(pm_gl['IPM']) * numSbjCmb_gl
    lenDf_overal  = combntn_ui * numModels_gl * len(pm_gl['OPM']) 

    # PATHS
    pathgrpLkUp = f'{pathOutput_ui}\\{foldername}\\View1\\SavingGroups'
    
    # Read combination group
    grpsLkUp = read_excel(f'{pathgrpLkUp}\\groupLkUp.xlsx').drop('Unamed: 0', axis=1)
    lstGrps  = list(grpsLkUp.Subjects)

    scores_single  = DataFrame(full((lenDf_single,len(colName_single_gl)), nan),columns=colName_single_gl).astype(dType_sngl_gl)
    scores_overall = DataFrame(full((lenDf_overal,len(colName_overal_gl)), nan),columns=colName_overal_gl).astype(dType_ovr_gl)
    rw_sng = 0
    rw_all = 0
    errLogCntr = -1
    errorLog   = dict()
    for i_grp, grpsbjs in zip(range(combntn_ui),lstGrps):
        pathGrp = f'G{i_grp}'

        for i_dic, (mdTyp, mdLst) in enumerate(numModels_gl.items()):
            for mdl in mdLst:
                pathData = f'{pathOutput_ui}\\{foldername}\\{pathGrp}\\entire\\Predict\\Perf\\Best\\{mdl}'

                try:
                    with open(f'{pathData}\\Bestscores.pckl','rb') as fp1:
                        data = read_pickle(fp1)
                    # with
                    for i_sb, SbID in enumerate(eval(grpsbjs)):
                        for i_PM, name_IPM in enumerate(pm_gl['IPM']):
                            scores_single.loc[rw_sng + i_PM,colName_single_gl]   = [mdTyp,mdl,SbID,pathGrp,name_IPM,data['single'][name_IPM][i_sb]]
                        # for
                        rw_sng += i_PM + 1
                    # for
                except Exception as e:
                    errLogCntr = errLogCntr + 1                
                    errorLog[errLogCntr] = f'Single,{pathGrp},{mdl},{e}]'
                # try

                try:
                    # Overall scores
                    for i_PM, name_OPM in enumerate(pm_gl['OPM']):
                        scores_overall.loc[rw_all + i_PM,colName_overal_gl]  = [mdTyp,mdl,pathGrp,name_OPM,data['overal'][name_OPM]]
                    #
                    rw_all += i_PM + 1
                except Exception as e:
                    errLogCntr = errLogCntr + 1                
                    errorLog[errLogCntr] = f'Overall,{pathGrp},{mdl},{e}]'
                # try
            # for mdl
        # for i_dic
    # for i_grp

    return scores_single, scores_overall, errorLog
#
