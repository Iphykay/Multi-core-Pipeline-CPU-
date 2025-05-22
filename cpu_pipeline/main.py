module_name = 'Parallel_Computing(CPU Based)'

'''
Version: v1.3.1

Description:
   the process of breaking down larger problems 
   into smaller, independent, often similar parts 
   that can be executed simultaneously by multiple 
   processors communicating via shared memory

Authors:
    Dr. Omer Muhammet Soysal
    Iphy Kelvin

Date Created     : 08/01/2024
Date Last Updated: 09/24/2024

Doc:
    <***>

Notes:
    <***>

ToDo:
event_code[0] must be modified to take care the event_code in config.py has more than one item.
See "this loop and the loop for executor will contradic."
'''

# CUSTOM IMPORTS
from config          import ConfigExtrctSoi as CfgExt, ConfigMain as CfgMain
from loader          import extractSoI, preProcessBeforeBand, transform_to_TopoMap, spatialTemp_map, pullTrainTest
from network_rcs     import dcnn, ae
from stats_generator import stats_gnrtr, transform_data
from rc_models       import KNN, RF, SVM, ANN 
from ui              import All_ui, LoadPrepare_ui as LdPrep_ui, SplitShufl as SS_ui, Dcnn_ui, allmodelsprm
from util            import save_all_hyperpmtrs, style_dataframe, get_info

# OTHER IMPORTS
import concurrent.futures
from multiprocessing import Manager
from time            import perf_counter
from numpy           import array, round, floor
from pandas          import read_pickle, DataFrame, ExcelWriter
from pickle          import dump
from types           import MappingProxyType
from os              import makedirs as osMakedirs, environ, getpid, getppid
import os
environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
environ['TF_CPP_MIN_LOG_LEVEL']  = '1'

import tensorflow as tf


# USER INTERFACE
combntn_ui       = All_ui['combntn_ui']
bandToUse_ui     = LdPrep_ui['bandToUse_ui']
FGLst_ui         = (dcnn,ae)                      #  All_ui['ntwkClsNameLst_ui'] # (dcnn)  # not in use for now Issue Pending
foldIDLst_ui     = All_ui['foldIDLst_ui']
isFGRC_ui        = All_ui['isFGRC_ui']
pathOutput_ui    = All_ui['pathOutput_ui']
pathInput_ui     = All_ui['pathInput_ui']
stmlusName_ui    = All_ui['stmlusName_ui']
loadpreprcss_ui  = All_ui['loadpreprcss_ui']
get_statFeats_ui = All_ui['get_feats_ui']
useStdMontage_ui = LdPrep_ui['useStdMontage_ui']
sizeX_ui         = All_ui['sizeX_ui']
sizeY_ui         = All_ui['sizeY_ui']
main_str_ui      = SS_ui['main_str_ui']
num_fold_ui      = All_ui['num_fold_ui']
nameRCLst_ui     = All_ui['nameRCLst_ui']
nameFGLst_ui     = All_ui['nameFGLst_ui'] 
seed_num_ui      = SS_ui['seed_num_ui']
trn_str_ui       = SS_ui['trn_str_ui']
tst_str_ui       = SS_ui['tst_str_ui']
val_size_ui      = Dcnn_ui['val_size_ui']
train_size_ui    = Dcnn_ui['train_size_ui']
viewIDLst_ui     = LdPrep_ui['viewIDLst_ui']
RCLst_ui         = (KNN,RF,SVM,ANN)  # KNN,RF,SVM,ANN
queryVal_ui      = All_ui['queryVal_ui']
useViewModel_ui  = LdPrep_ui['useViewModel_ui']
stimulus_ui      = All_ui['stmlusName_ui']
ViewUsed_ui      = LdPrep_ui['ViewUsed_ui']
cSubjects_ui     = All_ui['cSubjects_ui']

# CONSTANTS
dataSetNameLst_gl = CfgMain['DataSetName_gl']
colmnGrpBy1st_gl  = CfgExt['colmnGrpBy1st_gl'] 
colmnGrpBy2nd_gl  = CfgExt['colmnGrpBy2nd_gl']
isPrintTest_gl    = CfgMain['isPrintTest_gl']
numCores_gl       = CfgMain['numCores_gl'] - 1
timeinsecs_gl     = CfgMain['timeinsecs_gl']

# INITIALIATIONS
ModelRC       = {}
ModelFGRCSF   = {}
ModelFGRCFLD  = {}


class MultiProcessing:
    def __init__(self):
        '''
        Multiprocessing houses methods used in classifyiying 
        different stimulus using machine learning techniques
        '''
        self.inputData      = dict()
        self.df_trnsfrmd    = Manager().dict()
        self.datasetInfo    = dict()
        self.inputDataAll   = Manager().dict()
        self.datasetInfoAll = Manager().dict()
        self.combntn        = combntn_ui
        self.ModelFG        = Manager().dict()
        self.rsltQueryAll   = Manager().dict()
        self.rsltQuery_df   = dict()
    #

    def pullData_comb(self, bandID):
        """
        Pulls out the data for each combination of subjects
        in a semester or semesters

        Input:
        -----
        bandID  the band Id used

        """
        for idComb, sbComb in enumerate(combntn_ui):            
            #NOTE: indexes of samples in df_info MUST match the indexes of input_data          !!!!!!!!!!!!!!!
            indx                                 = dict(self.datasetInfoAll)[bandID]['subject'].isin(sbComb)
            self.inputData[f'{bandID}_{idComb}'] = dict(self.inputDataAll)[bandID][indx]

            self.datasetInfo[f'{bandID}_{idComb}'] = dict(self.datasetInfoAll)[bandID].loc[indx]
            self.rsltQuery_df[f'{bandID}_{idComb}'] = dict(self.rsltQueryAll)[bandID].reset_index().drop(columns=['index']).loc[indx]

            # Get the unique numbers
            unq_num = self.datasetInfo[f'{bandID}_{idComb}']['class_labels'].unique()

            # Apply change
            self.datasetInfo[f'{bandID}_{idComb}']['class_labels'] = self.datasetInfo[f'{bandID}_{idComb}']['class_labels'].map(dict(zip(sorted(unq_num), range(len(unq_num)))))
            self.datasetInfo[f'{bandID}_{idComb}']['class_labels'] = self.datasetInfo[f'{bandID}_{idComb}']['class_labels'].astype(int)

            if  isPrintTest_gl: print(f'input_data:\n {self.inputData[f"{bandID}_{idComb}"]}')
            if  isPrintTest_gl: print(f'df_info:\n{self.datasetInfo[f"{bandID}_{idComb}"]}')
        #
        if  isPrintTest_gl: print(self.inputDataAll)
    #

    def load_preprocess_data(self, bandID, viewID=None):
        '''
        Preprocesses the stimulus

        Input:
        -----
        bandID : names of band 
        '''
        
        print('Load data function started....', flush=True)
        # READ THE LOADER
        self.rawMneAll, self.infoSoiPart_all, self.rsltQueryAll[bandID], self.loc_ch, self.totSoI = extractSoI(stmlusName_ui,colmnGrpBy1st_gl,colmnGrpBy2nd_gl,
                                                                                                               queryVal_ui,useStdMontage_ui)

        print('Load data function ended...', flush=True)


        print(f'Peprocess data function started for {bandID}...', flush=True)
        dataset_info, soiAllFltrd, soiAllCndtn, soiAllFltrBand = preProcessBeforeBand(self.rawMneAll, self.infoSoiPart_all, bandID,
                                                                                      self.rsltQueryAll[bandID])
        
        if get_statFeats_ui:
            ## GENERATE FEATURES
            # #INPUT: SoI -> OUTPUT: Feature Vector
            features = stats_gnrtr(soiAllFltrBand, queryVal_ui, self.rsltQueryAll[bandID], bandID, self.totSoI)

            #INPUT: Feature Vector 2D: (num_Soi, num_features) -> OUTPUT: 3D (num_Soi, num_features, num_channels)
            self.df_trnsfrmd[bandID], self.datasetInfoAll[bandID] = transform_data(features)
            self.inputDataAll[bandID]                             = transform_to_TopoMap(self.df_trnsfrmd[bandID], sizeX_ui, sizeY_ui, 
                                                                                         self.df_trnsfrmd[bandID].shape[0], self.loc_ch)
        else:
            self.inputDataAll[bandID] = spatialTemp_map(soiAllFltrBand, viewID, bandID)
            rsltQuery                 = self.rsltQueryAll[bandID].loc[:,['semester','subject','session']].reset_index().drop(columns=['index'])
            rsltQuery['class_labels'] = [cSubjects_ui[rsltQuery.loc[:,'subject'][i]] for i in range(0,len(rsltQuery))] 

            self.datasetInfoAll[bandID] = rsltQuery

            del soiAllFltrBand
        # if

        print(f'Preprocess data function ended for {bandID}...', flush=True)
    #

    def save_data(self):
        '''
        Saves the feature map, dt_transformed and df_info
        '''
        if get_statFeats_ui:
            pathStatsData = f'{pathInput_ui}\\StatsData'; osMakedirs(pathStatsData, exist_ok=True)
            with (open(f'{pathStatsData}\\FeatMaps.pckl','wb') as pf1,
                  open(f'{pathStatsData}\\DataTransform.pckl','wb') as pf2,
                  open(f'{pathStatsData}\\DatasetInfo.pckl','wb') as pf3,
                  open(f'{pathStatsData}\\RsltQuery_df.pckl', 'wb') as pf4):
                dump(dict(self.inputDataAll), pf1)
                dump(dict(self.df_trnsfrmd), pf2)
                dump(dict(self.datasetInfoAll), pf3)
                dump(dict(self.rsltQueryAll), pf4)
        else:
            pathViewData = f'{pathInput_ui}\\ViewData'; osMakedirs(pathViewData, exist_ok=True)
            with (open(f'{pathViewData}\\ViewMaps.pckl','wb') as pf1,
                  open(f'{pathViewData}\\DatasetInfo.pckl','wb') as pf2,
                  open(f'{pathViewData}\\RsltQuery_df.pckl','wb') as pf3):
                dump(dict(self.inputDataAll), pf1)
                dump(dict(self.datasetInfoAll), pf2)
                dump(dict(self.rsltQueryAll), pf3)
        # if
    #

    def getData(self, viewID, savedatasets, bandID):
        '''
        Splits the data into training, validation and testing

        Input:
        -----
        viewID      : what view 
        savedatasets: saving what data
        bandID      : what band is used
        '''
        # DICT
        self.bandData = dict()

        for idComb, sbComb in enumerate(combntn_ui):
            pathGrp = f'G{idComb}'
            
            inputData = self.inputData[f'{bandID}_{idComb}']

            print(f'Data splitting started for {idComb} / {bandID}....', flush=True)
            self.bandData[f'{bandID}_{idComb}'] = {'XTrn': None, 'YTrn': None, 'XTst': None, 'YTst': None, 'XVal': None,
                                                   'YVal': None, 'RsltQTrain': None, 'fTrainOrg': None, 'cTrainOrg': None}
            
            (fTrn, cTrn), (fVal, cVal), (fTst, cTst), (rsltQueryTrn,fTrainOrg,cTrainOrg) = pullTrainTest(bandToUse_ui[0], inputData, self.datasetInfo[f'{bandID}_{idComb}'], 
                                                                                                         self.rsltQuery_df[f'{bandID}_{idComb}'], main_str_ui, trn_str_ui, tst_str_ui, 
                                                                                                         seed_num_ui, viewID=viewID)
            
            self.bandData[f'{bandID}_{idComb}']['XTrn']   = fTrn
            self.bandData[f'{bandID}_{idComb}']['YTrn']   = cTrn
            self.bandData[f'{bandID}_{idComb}']['XVal']   = fVal
            self.bandData[f'{bandID}_{idComb}']['YVal']   = cVal
            self.bandData[f'{bandID}_{idComb}']['XTst']   = fTst
            self.bandData[f'{bandID}_{idComb}']['YTst']   = cTst
            self.bandData[f'{bandID}_{idComb}']['RsltQTrain']  = rsltQueryTrn
            self.bandData[f'{bandID}_{idComb}']['fTrainOrg']   = fTrainOrg
            self.bandData[f'{bandID}_{idComb}']['cTrainOrg']   = cTrainOrg

            if savedatasets:
                if useViewModel_ui: 
                    pathData = f'{pathInput_ui}\\View{viewID}\\{pathGrp}\\{bandID}\\Datasets'; osMakedirs(pathData, exist_ok=True)
                else:
                    pathData = f'{pathInput_ui}\\{pathGrp}\\{bandID}\\Datasets'; osMakedirs(pathData, exist_ok=True)
                # if

                allDatasets = [(fTrn, cTrn), (fVal, cVal), (fTst, cTst)]
                with open(f'{pathData}\\alldatasets_{bandID}_{idComb}.pckl','wb') as pickle_file:
                    dump(allDatasets, pickle_file)
                # with
            # if
            print(f'Data splitting ended for {idComb} / {bandID}....', flush=True)
        # for idComb
    #

    def rc_models(self, idComb, ID, mdlClsRc, mdlNameRc, bandID, savebndtime, 
                  savebndhpt, savebndhpt1, trckPrsLkUp, sbComb):
        '''
        This function uses the traditional machine learning models to 
        classify stimulus.

        Input:
        ------
        mdlNameRc      : regular classiify names
        i_mdl_lrnr     : function learner for each regular clasiifier
        j_mdloutp      : function output for each regular classifier
        bandID         : name of the band; it is kept for future use
        comBID         : Combination ID to pull up subjects in a combination
        savebndtime    : dataframe to save the time taken for each model to finish
        savebndhpt     : dataframe to save the hyperparameters 
        savebndhpttyp1 : dataframe to save the hyperparameters for the cross
                         validation type 1
        trckPrsLkUp    : used to track the activity
        pathGrp        : name of groups of subjects
        '''
        print(f'Model analysis for regular classififers {mdlNameRc} started....', flush=True)

        # Start time
        startTime = perf_counter()

        ModelRC[f'{mdlNameRc}_{idComb}'] = mdlClsRc(ID, self.bandData[f'{bandID}_{idComb}']['XTrn'], array(self.bandData[f'{bandID}_{idComb}']['YTrn']), self.bandData[f'{bandID}_{idComb}']['XVal'], 
                                                    array(self.bandData[f'{bandID}_{idComb}']['YVal']), self.bandData[f'{bandID}_{idComb}']['XTst'], array(self.bandData[f'{bandID}_{idComb}']['YTst']), bandID, 
                                                    num_fold_ui, train_size_ui, val_size_ui, self.bandData[f'{bandID}_{idComb}']['RsltQTrain'], self.bandData[f'{bandID}_{idComb}']['fTrainOrg'], 
                                                    self.bandData[f'{bandID}_{idComb}']['cTrainOrg'], None, None, None, None, sbComb, runAll='All',foldIDLst_ui=['SF']+foldIDLst_ui, foldID=None)
        
        print(f'keyTrackProc ...',flush = True)
        try:
            trckPrsLkUp[f'{os.getpid()}_{os.getppid()}_{idComb}'] = [mdlNameRc, bandID, idComb]
            print(f'\n Track Process for : \n\t {mdlNameRc}-{bandID}-{idComb}\n\t trackProcessLookUp: {trckPrsLkUp}', flush=True)
        except Exception as e:
            print(f'Exception in trackProcessLookUp: \n\t{e}', flush=True)
        # try
        bedTime = perf_counter()

        # Time Elapsed 
        savebndtime[f'{bandID}_{idComb}_{mdlNameRc}'] = {f'{mdlNameRc}':round((bedTime-startTime)/timeinsecs_gl,2)}

        savebndhpt[f'{bandID}_{idComb}_{mdlNameRc}']  = {f'{mdlNameRc}':str(ModelRC[f'{mdlNameRc}_{idComb}'].PrmtrBest['SF'].values)}
        savebndhpt1[f'{bandID}_{idComb}_{mdlNameRc}'] = {f'{mdlNameRc}':str(ModelRC[f'{mdlNameRc}_{idComb}'].bestPrmtrFldRC.values)}

        print(f'Model analysis for regular classififers {mdlNameRc} ended....', flush=True)
    #

    def train_networkModels(self, idComb, ID, mdlFG, mdlNameFG, bandID, sbComb, trckPrsLkUp, *arg):
        
        print(f'Function Train {mdlNameFG} started for {idComb}/{ID}...', flush=True)

        # Start time
        startTime1 = perf_counter()

        errLogDlrn = arg[0]
        try:
            errLogDlrn[f'{idComb}_{mdlNameFG}'] = ''
            self.ModelFG[f'{mdlNameFG}_{idComb}'] = mdlFG(ID, self.bandData[f'{bandID}_{idComb}']['XTrn'], self.bandData[f'{bandID}_{idComb}']['YTrn'], self.bandData[f'{bandID}_{idComb}']['XVal'], 
                                                        self.bandData[f'{bandID}_{idComb}']['YVal'], self.bandData[f'{bandID}_{idComb}']['XTst'], array(self.bandData[f'{bandID}_{idComb}']['YTst']), 
                                                        bandID, num_fold_ui, train_size_ui, val_size_ui, self.bandData[f'{bandID}_{idComb}']['RsltQTrain'], self.bandData[f'{bandID}_{idComb}']['fTrainOrg'], 
                                                        self.bandData[f'{bandID}_{idComb}']['cTrainOrg'], sbComb, mdlNameRc=None, mdlClsRc=None, runAll='SF', foldIDLst_ui=['SF']) 
        except Exception as e:
            print(f'Exception in Network Models {idComb}_{mdlNameFG} during training:\n\t {e}', flush=True)
            errLogDlrn[f'{idComb}_{mdlNameFG}'] = e
        # try

        try:
            trckPrsLkUp[f'{os.getpid()}_{os.getppid()}_{idComb}'] = [f'{mdlNameFG}', bandID, idComb]
            print(f'\n Track Process for : \n\t {mdlNameFG},{bandID}_{idComb}\n\t trackProcessLookUp: {trckPrsLkUp}', flush=True)
        except Exception as e:
            print(f'Exception in trackProcessLookUp: \n\t{e}', flush=True)
        # try
        
        print(f'Function Train {mdlNameFG} ended for {idComb}/{ID}...', flush=True)
        

    def network_models(self, idComb, ID, mdlFG, mdlNameFG, bandID, savebndtime, savebndhpt, 
                       savebndhpt1, trckPrsLkUp, mdlClsRc, mdlNameRc, sbComb, *arg):
        # Start time
        startTime1    = perf_counter()
        nrtworkPrmtrs = dict()

        # Get the Parameters for netork model
        nrtworkPrmtrs[f'{idComb}_{mdlNameFG}'] = dict(self.ModelFG)[f'{mdlNameFG}_{idComb}'].PrmtrBest['SF'].values
        
        errLogDvSt = arg[0]
        try:
            errLogDvSt[f'Dv_{idComb}_{mdlNameFG}_{mdlNameRc}'] = ''
            ModelFGRCSF[f'{mdlNameFG}_{mdlNameRc}_{idComb}'] = mdlClsRc(dict(self.ModelFG)[f'{mdlNameFG}_{idComb}'].ID, dict(self.ModelFG)[f'{mdlNameFG}_{idComb}'].fTrnRepr, array(self.bandData[f'{bandID}_{idComb}']['YTrn']),
                                                                        dict(self.ModelFG)[f'{mdlNameFG}_{idComb}'].fValRepr, self.bandData[f'{bandID}_{idComb}']['YVal'], dict(self.ModelFG)[f'{mdlNameFG}_{idComb}'].fTstRepr, 
                                                                        self.bandData[f'{bandID}_{idComb}']['YTst'], bandID, num_fold_ui, train_size_ui, val_size_ui, self.bandData[f'{bandID}_{idComb}']['RsltQTrain'], 
                                                                        self.bandData[f'{bandID}_{idComb}']['fTrainOrg'], self.bandData[f'{bandID}_{idComb}']['cTrainOrg'], True, mdlNameFG, mdlNameRc,
                                                                        dict(self.ModelFG)[f'{mdlNameFG}_{idComb}'].reprLyr, sbComb, runAll='SF', foldIDLst_ui=['SF'], foldID=None)
        except Exception as e:
            print(f'Exception in FG+RC {idComb}_{mdlNameFG}_{mdlNameRc}:\n\t {e}', flush=True)
            errLogDvSt[f'Dv_{idComb}_{mdlNameFG}_{mdlNameRc}'] = e
        # try
        
        # Export it
        ModelFGRCSF[f'{mdlNameFG}_{mdlNameRc}_{idComb}'].export(dataSetNameLst_gl)
        
        # Cross Validation
        try:
            errLogDvSt[f'St_{idComb}_{mdlNameFG}_{mdlNameRc}'] = ''
            ModelFGRCFLD[f'{mdlNameFG}_{mdlNameRc}_{idComb}'] = mdlFG(ID, self.bandData[f'{bandID}_{idComb}']['XTrn'], self.bandData[f'{bandID}_{idComb}']['YTrn'], self.bandData[f'{bandID}_{idComb}']['XVal'], 
                                                                      self.bandData[f'{bandID}_{idComb}']['YVal'], self.bandData[f'{bandID}_{idComb}']['XTst'], array(self.bandData[f'{bandID}_{idComb}']['YTst']), 
                                                                      bandID, num_fold_ui, train_size_ui, val_size_ui, self.bandData[f'{bandID}_{idComb}']['RsltQTrain'], self.bandData[f'{bandID}_{idComb}']['fTrainOrg'], 
                                                                      self.bandData[f'{bandID}_{idComb}']['cTrainOrg'], sbComb, mdlNameRc, mdlClsRc, runAll='FLD',foldIDLst_ui=foldIDLst_ui)
        except Exception as e:
            print(f'Exception in CrossValidation {idComb}_{mdlNameFG}_{mdlNameRc}:\n\t {e}', flush=True)
            errLogDvSt[f'St_{idComb}_{mdlNameFG}_{mdlNameRc}'] = e
        # try

        print(f'keyTrackProc ...',flush = True)
        try:
            trckPrsLkUp[f'{os.getpid()}_{os.getppid()}_{idComb}'] = [f'{mdlNameFG}_{mdlNameRc}', bandID, idComb]
            print(f'\n Track Process for : \n\t {mdlNameFG},{bandID}_{idComb}\n\t trackProcessLookUp: {trckPrsLkUp}', flush=True)
        except Exception as e:
            print(f'Exception in trackProcessLookUp: \n\t{e}', flush=True)

        bedTime1 = perf_counter()
        
        # Saving Tine Elapsed
        savebndtime[f'{bandID}_{idComb}_{mdlNameFG}_{mdlNameRc}'] = {f'{mdlNameFG}_{mdlNameRc}':round((bedTime1 - startTime1)/timeinsecs_gl,2)}

        # Hyperparameters
        savebndhpt[f'{bandID}_{idComb}_{mdlNameFG}_{mdlNameRc}']  = {f'{mdlNameFG}_{mdlNameRc}':str(nrtworkPrmtrs[f'{idComb}_{mdlNameFG}']) + str(ModelFGRCSF[f'{mdlNameFG}_{mdlNameRc}_{idComb}'].PrmtrBest['SF'].values)}
        try:
            errLogDvSt[f'Pm_{idComb}_{mdlNameFG}_{mdlNameRc}'] = ''
            savebndhpt1[f'{bandID}_{idComb}_{mdlNameFG}_{mdlNameRc}'] = {f'{mdlNameFG}_{mdlNameRc}':str(ModelFGRCFLD[f'{mdlNameFG}_{mdlNameRc}_{idComb}'].bstPrmtrFldFG.values) + str(ModelFGRCFLD[f'{mdlNameFG}_{mdlNameRc}_{idComb}'].bstPrmtrFldRC.values)}
        except Exception as e:
            print(f'Exception in saving HPT1  {idComb}_{mdlNameFG}_{mdlNameRc}:\n\t {e}', flush=True)
            errLogDvSt[f'Pm_{idComb}_{mdlNameFG}_{mdlNameRc}'] = e
        # try

        print(f'Model analysis for regular classififers {mdlNameFG} ended....', flush=True)
    #


def trackProcess(fLst):
    for future in fLst:
        try:
            return [future.cancelled(), future.exception()]
        except Exception as e:
            print(f'Error trackProcess:\n\t{e}', flush=True)
            return [None, None]
        #
    # 
#  

def main():
    # Get the info in config and UI
    get_info(All_ui['expId']) 

    multiPrcess = MultiProcessing()

    pid  = getpid()
    ppid = getppid()

    strtTime_main = perf_counter()

    if loadpreprcss_ui:

        with concurrent.futures.ProcessPoolExecutor() as ldprcs_executor:
            if useViewModel_ui: useView = ViewUsed_ui
            else: useView = None
            loadpreprcs = [ldprcs_executor.submit(multiPrcess.load_preprocess_data, bandID, useView)
                           for bandID in bandToUse_ui]
            
            print('Waiting for result.....', flush=True)
            concurrent.futures.wait(loadpreprcs, timeout=60)
        # with

        multiPrcess.save_data()

    else:
        pathStatsData = f'{pathInput_ui}\\StatsData'
        pathSTData    = f'{pathInput_ui}\\ViewData'
        
        if get_statFeats_ui:
            with (open(f'{pathStatsData}\\FeatMaps.pckl','rb') as pf1,
                open(f'{pathStatsData}\\DataTransform.pckl','rb') as pf2,
                open(f'{pathStatsData}\\DatasetInfo.pckl','rb') as pf3,
                open(f'{pathStatsData}\\RsltQuery_df.pckl','rb') as pf4):
                    multiPrcess.inputDataAll   = read_pickle(pf1)
                    multiPrcess.df_trnsfrmd    = read_pickle(pf2)
                    multiPrcess.datasetInfoAll = read_pickle(pf3)
                    multiPrcess.rsltQueryAll   = read_pickle(pf4)
            # with
        else:
            with (open(f'{pathSTData}\\ViewMaps.pckl','rb') as pf1,
                  open(f'{pathSTData}\\DatasetInfo.pckl','rb') as pf2,
                  open(f'{pathSTData}\\RsltQuery_df.pckl','rb') as pf3):
                multiPrcess.inputDataAll   = read_pickle(pf1)
                multiPrcess.datasetInfoAll = read_pickle(pf2)
                multiPrcess.rsltQueryAll   = read_pickle(pf3)
            # with
        # if
    # if

    timeToProcessLoad = perf_counter() - strtTime_main

    # if numCores > len(combntn): lenChunk = int(np.ceil(len(combntn)/2))
    lenChunk = {'rc': max(1,int(floor(numCores_gl/len(nameRCLst_ui)))),
                'dm': max(1, int(floor(numCores_gl/len(nameFGLst_ui)))),
                'dr': max(1,int(floor(numCores_gl/(len(nameFGLst_ui) * len(nameRCLst_ui)))))}

    
    cmbnChnks = {key: [combntn_ui[i:i + lenChk] for i in range(0,len(combntn_ui),lenChk)] for key,lenChk in lenChunk.items()}

    for bandID in bandToUse_ui:
        savebndtime  = Manager().dict()
        savebndhpt   = Manager().dict()
        savebndhpt1  = Manager().dict()
        trckPrsLkUp  = Manager().dict()
        trackPro     = Manager().dict()
        errLogDlrn   = Manager().dict()
        errLogDvSt   = Manager().dict()

        # Pulls the data for each group
        multiPrcess.pullData_comb(bandID)

        # Training, validation and Testing datasets
        if useViewModel_ui: multiPrcess.getData(ViewUsed_ui, True, bandID) #
        else: multiPrcess.getData(None, True, bandID) #
         

        # # Regular Classifiers
        # idComb = -1
        # rc_anlys_cmb = dict()

        # # multiPrcess.rc_models(idComb, pathGrp,KNN,'KNN',bandID,savebndtime, savebndhpt, savebndhpt1, trckPrsLkUp, ('sb328','sb381'))
        # print('ProcessPoolExecutor for RC MODELS........', flush=True)
        # for i_ck, chnk in enumerate(cmbnChnks['rc']):
        #     with concurrent.futures.ProcessPoolExecutor() as executor_RC:
        #         for idCmbCh, sbComb in enumerate(chnk):
        #             print(f'{idCmbCh}/ {i_ck}: {sbComb} begins...', flush=True)
        #             idComb += 1
        #             if useViewModel_ui: pathGrp = f'{ViewUsed_ui}\\G{idComb}' 
        #             else: pathGrp = f'G{idComb}'
        #             rc_anlys = [executor_RC.submit(multiPrcess.rc_models, idComb, pathGrp, mdlClsRc, mdlNameRc, bandID, savebndtime, 
        #                                            savebndhpt, savebndhpt1, trckPrsLkUp, sbComb)
        #                         for mdlClsRc,mdlNameRc in zip(RCLst_ui,nameRCLst_ui)]
        #             rc_anlys_cmb[idComb] = rc_anlys
        #         # for 

        #         print('Waiting for result......', flush=True)
        #         concurrent.futures.wait(rc_anlys,timeout=60)

        #         #Keep track of processes
        #         trackPro[f'regcls_anlys_{i_ck}'] = trackProcess(rc_anlys)
        #     # with
        # # for

        idComb = -1
        ntwrk_lrnrs = dict()
        print('ProcessPoolExecutor for Network....', flush=True)
        for i_ck, chnk in enumerate(cmbnChnks['dm']):
            with concurrent.futures.ProcessPoolExecutor() as executor_FG:
                for idCmbCh, sbComb in enumerate(chnk):
                    print(f'{idCmbCh}/ {i_ck}: {sbComb} begins...', flush=True)
                    idComb += 1
                    if useViewModel_ui: pathGrp = f'{ViewUsed_ui}\\G{idComb}'
                    else: pathGrp = f'G{idComb}'
                    ntwrk = [executor_FG.submit(multiPrcess.train_networkModels, idComb, pathGrp, mdlFG, mdlNameFG, bandID, sbComb, 
                                                trckPrsLkUp, errLogDlrn)
                             for mdlFG, mdlNameFG in zip(FGLst_ui, nameFGLst_ui)]
                    ntwrk_lrnrs[idComb] = ntwrk
                # for

                print('Waiting for result......', flush=True)
                concurrent.futures.wait(ntwrk,timeout=60)

                #Keep track of processes
                trackPro[f'nt_anlys_{i_ck}'] = trackProcess(ntwrk)
            # with
        # for

        # Network + Rcs
        idComb = -1
        fg_anlys_cmb = dict()
        print('ProcessPoolExecutor for FG+RC MODELS........', flush=True)
        for i_ch, chnk in enumerate(cmbnChnks['dr']):
            with concurrent.futures.ProcessPoolExecutor() as executor_FGRC:
                for idCmbCh, sbComb in enumerate(chnk):
                    print(f'{idCmbCh}/ {i_ch}: {sbComb} begins...', flush=True)
                    idComb += 1
                    if useViewModel_ui: pathGrp = f'{ViewUsed_ui}\\G{idComb}'
                    else: pathGrp = f'G{idComb}'
                    fg_anlys = [executor_FGRC.submit(multiPrcess.network_models,idComb, pathGrp, mdlFG, mdlNameFG, bandID, savebndtime, 
                                                     savebndhpt, savebndhpt1, trckPrsLkUp, mdlClsRc, mdlNameRc, sbComb, errLogDvSt)
                                for mdlFG, mdlNameFG in zip(FGLst_ui, nameFGLst_ui) for mdlClsRc,mdlNameRc in zip(RCLst_ui,nameRCLst_ui)] 
                    fg_anlys_cmb[idComb] = fg_anlys
                # for

                print('Waiting for result......', flush=True)
                concurrent.futures.wait(fg_anlys,timeout=60)

                #Keep track of processes
                trackPro[f'fg_anlys_{i_ch}'] = trackProcess(fg_anlys)
            # with
        # for

        for idComb in range(0, len(combntn_ui)):
            if useViewModel_ui:
                pathOthers = f'{pathOutput_ui}\\View{ViewUsed_ui}\\G{idComb}\\Time'; osMakedirs(pathOthers, exist_ok=True)
            else:
                pathOthers = f'{pathOutput_ui}\\G{idComb}\\Time'; osMakedirs(pathOthers, exist_ok=True)

            try:
                timedict = {}
                for inme in list(dict(savebndtime)):
                    if inme.startswith(f'{bandID}_{idComb}'):
                        mdlbndtime = dict(savebndtime)[f'{inme}']

                        # Extract the keys and values
                        timedict.update(mdlbndtime)
                    # if
                # for

                savetime = DataFrame.from_dict(list(timedict.items()))
                savetime.columns = ['Model','Time Elapsed']
                with ExcelWriter(f'{pathOthers}\\{bandID}Modelruntime.xlsx') as writer_model:
                    savetime.to_excel(writer_model, sheet_name=f'{bandID}Runtimes')
                # with
            except Exception as e:
                print(f'Error in model_time:\n\t {e}', flush=True)
            # try

            try:
                hptdict = {}; hptdict1 = {}
                for ihpt in list(dict(savebndhpt)):
                    if ihpt.startswith(f'{bandID}_{idComb}'):
                        mdlhpt  = dict(savebndhpt)[f'{ihpt}']
                        mdlhpt1 = dict(savebndhpt1)[f'{ihpt}']

                        # Extract the keys and values
                        hptdict.update(mdlhpt); hptdict1.update(mdlhpt1)
                    # if
                # for

                # Put the extracted items into a dataframe
                savemdlhpt    = DataFrame.from_dict(list(hptdict.items()))
                savemdlhptty1 = DataFrame.from_dict(list(hptdict1.items()))
                savemdlhpt.columns = ['Model','BestParameters']; savemdlhptty1.columns=['Model','BestParameters']
                
                # Get the names of the models used
                mdlused = list(savemdlhpt['Model']); mdlusedtyp1 = list(savemdlhptty1['Model'])
                savehypmtrs     = save_all_hyperpmtrs(allmodelsprm, mdlused)
                savehypmtrstyp1 = save_all_hyperpmtrs(allmodelsprm, mdlusedtyp1)

                # Putting the data into the original dataframe
                for iM in mdlused:
                    savehypmtrs.iloc[mdlused.index(iM),1]     = savemdlhpt.iloc[mdlused.index(iM),1]
                    savehypmtrstyp1.iloc[mdlused.index(iM),1] = savemdlhptty1.iloc[mdlused.index(iM),1]
                #
                mainhypa     = style_dataframe(savehypmtrs)
                mainhypabst1 = style_dataframe(savehypmtrstyp1)
                with ExcelWriter(f'{pathOthers}\\{bandID}HPT.xlsx') as writerhpt:
                    mainhypa.to_excel(writerhpt, sheet_name=f'{bandID}HPT')
                with ExcelWriter(f'{pathOthers}\\{bandID}HPTTYP1.xlsx') as writerhptyp1:
                    mainhypabst1.to_excel(writerhptyp1, sheet_name=f'{bandID}HPTTYP1')
            except Exception as e:
                print(f'Error in saving the Hyperparameters: \n\t {e}', flush=True)
                try:
                    print(f'VALUES savehpt: {savemdlhpt.values()}', flush=True)
                    print(f'VALUES savehpt: {savemdlhptty1.values()}', flush=True)
                except Exception as e:
                    print(f'Error in printing VALUES savehpt:\n\t{e}', flush=True)
                # try
            # try

            try:
                if useViewModel_ui: pathsaveTrck = f'{pathOutput_ui}\\View{ViewUsed_ui}\\G{idComb}\\SavingTrck'; osMakedirs(pathsaveTrck, exist_ok=True)
                else: pathsaveTrck = f'{pathOutput_ui}\\G{idComb}\\SavingTrck'; osMakedirs(pathsaveTrck, exist_ok=True)
                with open(f'{pathsaveTrck}\\{stimulus_ui}_TrckPrsLkUp.txt','w') as f: f.write(str(trckPrsLkUp))

                with open(f'{pathsaveTrck}\\{stimulus_ui}_TrckPrsLkUp.csv','w') as f0:
                    f0.write(f'KEY,MODEL,GROUP\n')
                    for key,val in trckPrsLkUp.items(): f0.write(f'{key},{val[0]},{val[2]}\n')
                # with

                with open(f'{pathsaveTrck}\\{stimulus_ui}_Exceptions.txt','w') as f1: f1.write(str(trackPro))
            except Exception as e:
                print(f'Error writing Experiment.txt:\n\t{e}', flush=True)
            # try
        # for
    # for bandID

    # Sving Groups
    pathsaveGrps = f'{pathOutput_ui}\\SavingGroups'; osMakedirs(pathsaveGrps, exist_ok=True)
    grpLkup = DataFrame({'GroupID': list(range(len(combntn_ui))), 'Subjects': list(combntn_ui)})

    # LookUpTable
    try:
        with ExcelWriter(f'{pathsaveGrps}\\groupLkUp.xlsx') as writer:
            grpLkup.to_excel(writer, sheet_name='Groups')

        with open(f'{pathsaveGrps}\\groupLkUp.pckl','wb') as pickle_file:
            dump(grpLkup, pickle_file)            
    except Exception as e:
        print(f'Error in saving groupLkUp:\n\t{e}', flush=True)
    #

    try:
        pathsaveGrps = f'{pathOutput_ui}\\SavingChunks'; osMakedirs(pathsaveGrps, exist_ok=True)
        with open(f'{pathsaveGrps}\\Chunks.txt','w') as f1:
            f1.write(str(cmbnChnks))
    except Exception as e:
        print(f'Error in saving Chunks:\n\t{e}', flush=True)
    # try




if __name__ == '__main__':
    main()
    print('DONE', flush=True)




                







    











