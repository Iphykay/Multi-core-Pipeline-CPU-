module_name = 'dataAssembler_2Level'

'''
Version: v1.3.1

Description:
   Get the data used for visualization

Authors:
    Dr. Omer Muhammet Soysal
    Iphy Kelvin

Date Created     : 10/15/2024
Date Last Updated: 10/15/2024

Doc:


Notes:
    <***>
'''


# CUSTOM IMPORTS
from config_compile  import ConfigMainCmp
from ui              import All_ui
import visualization as vis

# OTHER IMPORTS
import pandas            as pd
import pickle            as pckl

# USER INTERFACE
pathOutput_ui = All_ui['pathOutput_ui']

# CONSTANTS
functnStat_gl = ConfigMainCmp['functnStat_gl']
folderStat_gl = ConfigMainCmp['folderStat_gl']
nameStat_gl   = ConfigMainCmp['nameStat_gl']

# INTIALIZATIONS
dataDf_grpL_1_gl   = {}
dataAllStatL_1_gl  = {}
dataAllStatL_2_gl  = {}


def dataAssembler_2Level(df, pathName, lstCatgryCH_0=list, colFltr_CH0=str, grpByL1=['Model','Subject'], 
                         grpByL2=['Subject'], colViz='Score', pathType=str):
    '''
    Assemble data for interpretations

    Input:
    -----
    df           : dataframe
    lstCatgryCH_0: List of categories at the concept-hierarchy level-0. E.g., lst_pm: list of performance measures
    colFltr_CH0  : Column to filter data at the concept-hierarchy level-0. E.g., pm_cat: 'PM'
    grpByL1      : Columns to apply group by at the 1st level
    grpByL2      : Columns to apply group by at the 2nd level
    colViz       :  Column for which to visualize numeric values
    pathName     : StatsRepr or ST Repr
    
    Output:
    ------
    Returns dataDf_grpL_1, dataAllStatL_1, dataAllStatL_2
    '''

    # Path to save the plots
    pathExport = pathName
    
    #valCol_ch0: pm_name, value from the column "concept-hierarchy level-0"
    for catgry_ch0 in lstCatgryCH_0:  #Category to visualize scores for (column for figure title)
        data_grpL_1 = df.loc[df[colFltr_CH0].isin([catgry_ch0])]

        for fnSt_1, nmSt_1 in zip(functnStat_gl,nameStat_gl):
 
            # data for each statistics
            dataDf_grpL_1_gl[catgry_ch0] = data_grpL_1

            # Statistics Level-1
            dataStL_1 = data_grpL_1.groupby(by=grpByL1)[colViz].apply(fnSt_1).reset_index()

            dataStL_2 = {}
            
            # Statistics Level-2
            for fnSt_2, nmSt_2 in zip(functnStat_gl,nameStat_gl):           
                try:
                    dataStL_2[nmSt_2] = dataStL_1.groupby(by=grpByL2)[colViz].apply(fnSt_2).reset_index()
                except Exception as e:
                    print(f'\n\t Exception: {catgry_ch0} {nmSt_1} {nmSt_2} of {pathExport} {pathType}. Most likely due to missing values !!!!!\n\tException: {e}')
                # try
            # for

            dataAllStatL_1_gl[f'{nmSt_1}_{catgry_ch0}'] = dataStL_1
            dataAllStatL_2_gl[f'{nmSt_1}_{catgry_ch0}'] = dataStL_2
        # for
    # for

    # Save the data
    try:
        with pd.ExcelWriter(f'{pathExport}\\dataGroupLevel_1.xlsx') as grpdata_wrtr:
            for catgry_ch0, data in zip(lstCatgryCH_0, dataDf_grpL_1_gl.values()):
                data.to_excel(grpdata_wrtr, sheet_name=f'{catgry_ch0}')
        # with

        with pd.ExcelWriter(f'{pathExport}\\dataAllStatL_1.xlsx') as statslvl1_wrtr:
            for keynames, data in zip(dataAllStatL_1_gl.keys(),dataAllStatL_1_gl.values()):
                data.to_excel(statslvl1_wrtr, sheet_name=f'{keynames}')
        # with

        with pd.ExcelWriter(f'{pathExport}\\dataAllStatL_2.xlsx') as statlvl2_wrtr:
            for key_L1, data_L1 in zip(dataAllStatL_2_gl.keys(), dataAllStatL_2_gl.values()):
                for key_L2, data_L2 in zip(data_L1.keys(), data_L1.values()):
                    data_L2.to_excel(statlvl2_wrtr, sheet_name=f'{key_L2}{key_L1}')
        #
    except:
        print(f'\n excel file not created in dataAssembler_2Level!', flush=True)
    #

    try:
        with open(f'{pathExport}\\dataGroupLevel_1.pckl','wb') as pickle_file:
            pckl.dump(dataDf_grpL_1_gl, pickle_file)

        with open(f'{pathExport}\\dataAllStatL_1.pckl','wb') as pickle_file:
            pckl.dump(dataAllStatL_1_gl, pickle_file)

        with open(f'{pathExport}\\dataAllStatL_2.pckl','wb') as pickle_file:
            pckl.dump(dataAllStatL_2_gl, pickle_file)
    except:
            print(f'\n pickle file not created in dataAssembler_2Level!')
    #

    return dataDf_grpL_1_gl, dataAllStatL_1_gl, dataAllStatL_2_gl
#


def plotAssembler(scoreAll, pmNameAll, pmCatgry, path):
    '''
    Function that plots the data distribution for performance meausres

    Input:
    ------
    scoreAll : dataframe with scores (list holding a dataframe)
    pmNameAll: list of performance meausre scores
    pmCatgry : Either IPM or OPM
    pathName : Fold or Best
    '''

    for score, pmName, pmCat in zip(scoreAll, pmNameAll, pmCatgry):
        if pmCat == 'PM':
            pathType = f'I{pmCat}'
            dataDf_grpL_1, dataAllStatL_1, dataAllStatL_2 = dataAssembler_2Level(score, path, pmName, pmCat, pathType=pathType)
        elif pmCat == 'OPM':
            dataDf_grpL_1, dataAllStatL_1, dataAllStatL_2 = dataAssembler_2Level(score, path, pmName, pmCat, grpByL1=['Model'], grpByL2=['Model'], 
                                                                                 pathType=pathType)
        # if

        data_L1 = [dataDf_grpL_1]
        for data_v in data_L1:
            vis_plot = vis.Visualizer(data_v, path, [pmCat], fldrName=None)
            vis_plot.violinplot()
        # for

        data_all_L1 = [dataAllStatL_1]
        for data_v, fname in zip(data_all_L1, folderStat_gl):
            vis_plot = vis.Visualizer(data_v, path, [pmCat], fldrName=fname) 
            vis_plot.barplot() 
        # for
    # for



