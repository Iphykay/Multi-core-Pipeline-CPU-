module_name = 'util'

'''
Version: v1.1.0

Description:
    Utility components

Authors:
    <***>

Date Created     :  02/28/2025
Date Last Updated:  02/28/2025

Doc:
    <***>

Notes:
    <***>
'''


# CUSTOM IMPORTS
from config import ConfigMain as CfgMain
from ui     import All_ui, LoadPrepare_ui as LdPrep_ui

# OTHER IMPORTS
from numpy                   import arctan2, sqrt, cos, sin, array, max, round, min, median, diag
from os                      import makedirs as osMakedirs, getcwd
from pickle                  import dump
from pandas                  import DataFrame
from shutil                  import copyfile as sh_cpyfile

# USER INTERFACE
bandToUse_ui    = LdPrep_ui['bandToUse_ui']
pathInput_ui    = f'{All_ui['pathInput_ui']}{bandToUse_ui[0]}'
pathOutput_ui   = f'{All_ui['pathOutput_ui']}'
useViewModel_ui = LdPrep_ui['useViewModel_ui']

# CONSTANTS
PI_DIV_2_GL  = CfgMain['PI_DIV_2_GL']
N_SPLITS     = All_ui['num_fold_ui']

def save_hyperpmtr(ID, data, foldID, bandToUse, name=str):
    '''
    Saves pickle files of the hyperparamters after
    tuning

    Input:
    -----
    viewID  : what view (1,2 or 3)
    data    : input data
    name    : name of the data saved

    Output:
    ------
    : Returns none
    '''
    
    if name[-1].isdigit(): # KNN0, SVM1
        add_name = name[:-1]
    else:
        add_name = name
    #
    if useViewModel_ui:
        pathHPT = f'{pathOutput_ui}\\View{ID}\\{bandToUse}\\Model\\{add_name}'; osMakedirs(pathHPT, exist_ok=True)
    else:
        pathHPT = f'{pathOutput_ui}\\{ID}\\{bandToUse}\\Model\\{add_name}'
    try:
        with open(f'{pathHPT}\\hpt_{name}_{foldID}.pckl','wb') as pickle_file:
            dump(data, pickle_file)
    except Exception as e:
        print(f'Error in exporting pickle file for best parameters:\n\t{e}', flush=True)
    #
#

def projection_XYZtoXY(pos): 
    '''
    Converts 3D coordinate to 2D coordinate

    Input:
    -----
    pos : positions in 3D  (x,y,z)

    Output:
    ------
    Returns the projected coordinate using the polar projection
    '''

    # 3D to Polar
    # x, y, z = pos[0], pos[1], pos[2]
    # radius = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2))
    azimuth_angle = arctan2(pos[1],pos[0]) 
    radian        = PI_DIV_2_GL - arctan2(pos[2],sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2))
    
    # Polar to Cartesian
    x_pos = radian*cos(azimuth_angle)
    y_pos = radian*sin(azimuth_angle)

    return x_pos, y_pos
#

def get_info(expId):
    '''
    Gets the variables in the User interface and Config py files

    Input:
    ------
    expId: experiment ID path

    Output:
    -------
    : Returns exported config and ui 
    '''
    pathInfo = f'{pathOutput_ui}\\{CfgMain['folder_info_gl']}'; osMakedirs(pathInfo, exist_ok=True)

    expId_only = expId.split('\\')[0]

    try:
        sh_cpyfile(f'{getcwd()}\\config.py',f'{pathInfo}\\config_{expId_only}.py')
        sh_cpyfile(f'{getcwd()}\\ui.py',f'{pathInfo}\\ui_{expId_only}.py')
    except Exception as e:
        print(f'Error in saving config.py:\n\t{e}',flush=True)
    # try
#

def find_min_median_max(data):
    '''
    Finds the minimum, median and maximum in the list
    '''
    min_med_max = array([round(min(data),2), round(median(data),2), round(max(data),2)])
    # for

    return min_med_max
#

def mathfunc(data, mathfun_name):
    """
    Outputs a mathematical function (minimum, median and maximum)
    for elements in arrays individually

    Input:
    -----
    data        : 1D vector
    mathfun_name: name of math function

    Output:
    ------
    : Return Min, median or max values
    """
    if mathfun_name == 'min':
        if data.shape == (N_SPLITS,):
            return round(min(data),2)
        else:
            return round(min(data, axis=1),2)
    elif mathfun_name == 'median':
        if data.shape == (N_SPLITS,):
            return round(median(data),2)
        else:
            return round(median(data, axis=1),2)
    else:
        if data.shape == (N_SPLITS,):
            return round(max(data),2)
        else:
            return round(max(data, axis=1),2)
#

def save_all_hyperpmtrs(allmodelsprm, modelsused):
    """
    This functions puts all hyperparamters into a dataframe

    Input:
    ------
    allmodelsprm: all models paramters used (dict)
    modelsused  : the models used 

    Output:
    ------
    : Return dataframe with hyperparameters
    """
    prmtrs = dict((k, allmodelsprm[k]) for k in modelsused if k in allmodelsprm)
    savehyprmtr = {'parameters':{'Hyperparameters':prmtrs,
                                 'BestParameters':{}}}
    
    out_savehypr = DataFrame(savehyprmtr['parameters'], dtype=object)

    return out_savehypr
#

def style_dataframe(df_name):
    """
    Styles the dataframe

    Input:
    ------
    df_name: name of the dataframe

    Output:
    ------
    : Return Styled dataframe
    """
    styler = df_name.style.set_properties(subset=['Hyperparameters','BestParameters'],
                                                     **{'width':'350px'}).set_table_styles([{'selector':'th.col_heading','props':'text-align:center'},
                                                                                            {'selector':'tr','props':'line-height:30px'}])
    return styler
#


def ratioCM(confmtrx):
    fp = confmtrx.sum(axis=0)-diag(confmtrx)
    fn = confmtrx.sum(axis=1)-diag(confmtrx)
    tp = diag(confmtrx)
    tn = confmtrx.sum() - (fp+fn+tp)

    # change the object type
    fp = fp.astype(float)
    fn = fn.astype(float)
    tn = tn.astype(float)

    return fp,fn,tp,tn
#