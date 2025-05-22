module_name = 'compile_folds'

'''
Version: v1.3.1

Description:
   Put all the fold score together for each group

Authors:
    Dr. Omer Muhammet Soysal
    Iphy Kelvin

Date Created     : 10/26/2024
Date Last Updated: 10/26/2024

Doc:


Notes:
    <***>
'''


# CUSTOM IMPORTS
from config_compile import ConfigMainCmp, ConfigPMCmp, ConfigModelsCmp
from ui             import All_ui


# OTHER IMPORTS
from ast    import literal_eval
from csv    import writer as csvW
from numpy  import full, nan, array
from pandas import read_excel, DataFrame, read_pickle
from pickle import dump
from os     import makedirs as osMakedirs


# USER INTERFACE
pathOutput_ui = All_ui['pathOutput_ui']


# CONSTANTS
colNamesFold_gl = ConfigMainCmp['colNames_fold_gl']
numFold_gl      = ConfigMainCmp['num_fold_gl']
models_gl       = ConfigModelsCmp['models_gl']
numModels_gl    = ConfigMainCmp['numModels_gl']
pm_gl           = ConfigPMCmp['pmName_gl']['ALL']
numSbjCmb_gl    = ConfigMainCmp['numSbjsAll_ui']



def folds_Assembler():
    '''
    Assembles all the fold data for eachgroup

    Input:
    ------
    foldername: StatsRepr or STRepr
    '''
    # PATHS
    pathgrpLkUp = f'{pathOutput_ui}\\SavingGroups'
    pathFolds   = f'{pathOutput_ui}\\'

    lenBlockPM   = numFold_gl * numSbjCmb_gl

    # Read combination group
    grpsLkUp = read_excel(f'{pathgrpLkUp}\\groupLkUp.xlsx').drop('Unnamed: 0', axis=1)
    lstGrps  = list(grpsLkUp.Subjects)

    # Number of rows
    lenDf = len(lstGrps)*numModels_gl*len(pm_gl)*lenBlockPM

    df_scrs  = DataFrame(full((lenDf,len(colNamesFold_gl)), nan), columns=colNamesFold_gl)
    df_scrs[['Group','Model','PM','Subject']] = ''

    # Read for each group
    errLogCntr = -1
    errorLog   = dict()    
    rw_bgn = 0; rw_end = numSbjCmb_gl - 1
    for idxGrp in range(len(lstGrps)):
        pathGrp = f'G{idxGrp}'
        sbjs    = array(list(literal_eval(lstGrps[idxGrp]))).reshape(1,numSbjCmb_gl)

        for i_dic, (mdTyp, mdLst) in enumerate(models_gl.items()):
            for mdl in mdLst:
                try:
                    with open(f'{pathFolds}{pathGrp}\\entire\\Predict\\Perf\\Best\\{mdl}\\Foldscores.pckl','rb') as fp:
                        fldr_scrs = read_pickle(fp)
                    # with

                    for pm in pm_gl:
                        for idx_fld in range(numFold_gl):
                            mdTypA  = array([mdTyp] * numSbjCmb_gl).reshape(1,numSbjCmb_gl).astype(str)
                            idxFld  = array([idx_fld] * numSbjCmb_gl).reshape(1,numSbjCmb_gl).astype(int)
                            pM      = array([pm] * numSbjCmb_gl).reshape(1,numSbjCmb_gl).astype(str)
                            mDl     = array([mdl] * numSbjCmb_gl).reshape(1,numSbjCmb_gl).astype(str)
                            grpName = array([pathGrp] * numSbjCmb_gl).reshape(1,numSbjCmb_gl).astype(str)
                            scrs    = fldr_scrs[pm][idx_fld]

                            df_scrs.loc[rw_bgn:rw_end,colNamesFold_gl[:-1]] = [mdTypA, grpName, mDl, pM, idxFld, sbjs]
                            df_scrs.loc[rw_bgn:rw_end,'Score']               = scrs

                            rw_bgn += numSbjCmb_gl; rw_end += numSbjCmb_gl
                        # for
                    # for pm
                except Exception as e:
                    errLogCntr = errLogCntr + 1
                    errorLog[errLogCntr] = f'Single, {pathGrp}, {mdl}, {pm}, at dataAssembler_folds \n\t {e}'
                    print(f'Exception: {errorLog[errLogCntr]}')
                # try
            # for
        # for
    # for idxGrp

    return df_scrs, errorLog
#
