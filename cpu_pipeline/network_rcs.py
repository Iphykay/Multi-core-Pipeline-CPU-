module_name = 'Network_with_RCs'

'''
Version: v1.0.0

Description:
    CNN

Authors:
    Dr. Omer Muhammet Soysal
    Iphy Kelvin

Date Created     : 12/23/2024
Date Last Updated: 12/28/2024

Doc:
    <***>a

Notes:
    <***>
'''


# CUSTOM IMPORTS
from config         import ConfigDcnnAe as CfgDcnnAe, ConfigFeatGen_HC as CfgFeatGen, ConfigMain as CfgMain
from network_tuner  import customTuner
from ui             import Dcnn_ui, num_init_pts_cnv2D_ui, All_ui,num_init_pts_ae_ui, Ae_ui, Rc_ui, LoadPrepare_ui as LdPrep_ui
from util           import save_hyperpmtr

# OTHER IMPORTS
from imblearn.metrics        import specificity_score, geometric_mean_score
from keras                   import Model
from keras.layers            import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Input, UpSampling2D
import matplotlib.pyplot     as plt
from numpy                   import argmax, nonzero, sqrt, floor, log10, array
from os                      import makedirs as osMakedirs, environ
environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
environ['TF_CPP_MIN_LOG_LEVEL']  = '1'
from pandas                  import DataFrame, Series, ExcelWriter
from pickle                  import dump
from rc_models               import KNN, RF, SVM, ANN 
from sklearn.metrics         import confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, fbeta_score, matthews_corrcoef, cohen_kappa_score, precision_score, recall_score, ConfusionMatrixDisplay
from seaborn                 import boxplot, violinplot, barplot, set_theme
from sklearn.model_selection import StratifiedShuffleSplit

# USER INTERFACE
# foldIDLst_ui      = All_ui['foldIDLst_ui']
useViewModel_ui   = LdPrep_ui['useViewModel_ui']

# FOR PLOT
fontSize_title_ui = Dcnn_ui['fontSize_title_ui']
fontSize_xy_ui    = Dcnn_ui['fontSize_xy_ui']
fontSize_lgnd_ui  = Dcnn_ui['fontSize_lgnd_ui']
nameRCLst_ui      = All_ui['nameRCLst_ui']
figSize_barplt_ui = Dcnn_ui['figSize_barplt_ui']
isExport_ui       = All_ui['isExport_ui']
isBinary_ui       = Rc_ui['isBinary_ui']
perMeasureLst_ui  = Dcnn_ui['perMeasurLst_ui']
rw_subplt_ui      = Dcnn_ui['rw_subplt_ui']
RCLst_ui          = (KNN,RF,SVM,ANN)
col_subplt_ui     = Dcnn_ui['col_subplt_ui']

# OTHERS
num_classes_ui    = All_ui['numSbjsCmd_ui']
num_fold_ui       = All_ui['num_fold_ui']
pathOutput_ui     = All_ui['pathOutput_ui']

# CONSTANTS
folder_best_gl         = CfgMain['folder_BstperScores_gl']
folder_bestplot_gl     = CfgMain['folder_BstperPlot_gl']
classPrdct_gl          = CfgMain['classPrdt_gl']
col_x2_gl              = CfgFeatGen['col_x2_gl'] 
col_y_gl               = CfgFeatGen['col_y_gl']  
folder_hptLog_gl       = CfgMain['folder_hptLog_gl']
folder_prdct_gl        = CfgMain['folder_prdct_gl']
folder_prjct_gl        = CfgMain['folder_prjct_gl']
folder_model_gl        = CfgMain['folder_model_gl']
folder_modelPlot_gl    = CfgMain['folder_modelPlot_gl']
folder_perScores_gl    = CfgMain['folder_perScores_gl']
folder_perPlot_gl      = CfgMain['folder_perPlot_gl']
ktObjective_gl         = CfgDcnnAe['ktObjective_gl']
keys_prdctPerf_gl      = CfgMain['keys_prdctPerf_gl']
LOG_10_2_GL            = CfgDcnnAe['LOG_10_2_GL']
dataSetNameLst_gl      = CfgMain['DataSetName_gl']
NUM_LAYRS_GL           = CfgDcnnAe['NUM_LAYRS_UI']
numBestModels_gl       = CfgMain['numBestModels_gl']
SIZE_LAST_LAYER_IMG_GL = CfgDcnnAe['SIZE_LAST_LAYER_IMG_UI']
scorePrdct_gl          = CfgMain['scorePrdct_gl']
scoreIndvdl_gl         = CfgMain['scoreIndvdl_gl']
scoreOvrall_gl         = CfgMain['scoreOvrall_gl']

# INITIALIZATION
# history   = {}
# model     = {}
# PrmtrBest = {}
# PrdctPerf = {}

class BaseML:
    def __init__(self, ID, bandToUse_ui, model_name, max_trial, num_init_pt, exetnPerTrial, sbComb, rsltQuery, foldIDLst_ui=list):
        self.allFold_scores = {}
        self.bandToUse_ui   = bandToUse_ui
        self.bstScore       = {}
        self.exetnPerTrial  = exetnPerTrial
        self.history        = {}
        self.model_name     = model_name
        self.Model          = {}
        self.max_trial      = max_trial
        self.ModelFGRC      = {}
        self.num_init_pt    = num_init_pt
        self.saveBest       = {}
        self.PrmtrBest      = {}
        self.PrmtrRC        = {}
        self.PrdctPerf      = {}
        self.PrdctPrefFGRC  = {}
        self.rsltQuery      = rsltQuery
        self.reprLyr        = {}
        self.sbComb         = sbComb
        self.foldIDLst_ui   = foldIDLst_ui
        if useViewModel_ui:
            self.ID       = f'View{ID}'
        else:
            self.ID       = f'{ID}'
        # if
        self.pathModelPrdct  = f'{pathOutput_ui}\\{self.ID}\\{self.bandToUse_ui}\\{folder_prdct_gl}\\{self.model_name}'
        self.pathPerfSngle   = f'{pathOutput_ui}\\{self.ID}\\{self.bandToUse_ui}\\{folder_perScores_gl}\\{self.model_name}'
        self.pathModel       = f'{pathOutput_ui}\\{self.ID}\\{self.bandToUse_ui}\\{folder_model_gl}\\{self.model_name}'
        self.pathModelPlot   = f'{pathOutput_ui}\\{self.ID}\\{self.bandToUse_ui}\\{folder_modelPlot_gl}\\{self.model_name}'
        self.pathPerfPlot    = f'{pathOutput_ui}\\{self.ID}\\{self.bandToUse_ui}\\{folder_perPlot_gl}\\{self.model_name}'
        self.pathBestPerf    = f'{pathOutput_ui}\\{self.ID}\\{self.bandToUse_ui}\\{folder_best_gl}\\{self.model_name}'
        self.pathBstPerfPlot = f'{pathOutput_ui}\\{self.ID}\\{self.bandToUse_ui}\\{folder_bestplot_gl}\\{self.model_name}'

        for nameDataSet in dataSetNameLst_gl:
            self.PrdctPerf[nameDataSet] = {}
            self.PrdctPrefFGRC[nameDataSet] = {}
            for foldID in foldIDLst_ui:
                self.PrdctPerf[nameDataSet][foldID] = {}
                self.PrdctPrefFGRC[nameDataSet][foldID] = {}
                for key in keys_prdctPerf_gl:
                    self.PrdctPerf[nameDataSet][foldID][key] = {}
                # for
            # for
        # for
        for namePerfScore in perMeasureLst_ui:
            self.allFold_scores[namePerfScore] = {}
        # for
    #

    def export(self, dataSetNameLst_gl, useFGRC=bool, mdlNameRc=None):
        '''
        Exports variables to Excel and pickle files.
        '''
        for nameDataSet in dataSetNameLst_gl:
            for i_fold, foldID in enumerate(self.foldIDLst_ui):

                # Path
                if mdlNameRc is not None:
                    pathPrdct       = f'{self.pathModelPrdct}_{mdlNameRc}\\{foldID}'; osMakedirs(pathPrdct, exist_ok=True)
                    pathFldSngle    = f'{self.pathPerfSngle}_{mdlNameRc}\\{foldID}'; osMakedirs(pathFldSngle, exist_ok=True) 
                    pathPerfIndiv   = f'{self.pathModel}_{mdlNameRc}\\{foldID}'; osMakedirs(pathPerfIndiv, exist_ok=True)
                    pathMdl         = f'{self.pathMdl}_{mdlNameRc}\\{foldID}'; osMakedirs(pathMdl, exist_ok=True)
                    pathModelPrdct  = f'{self.pathModelPrdct}_{mdlNameRc}'; osMakedirs(pathModelPrdct, exist_ok=True)
                    pathBestPerf    = f'{self.pathBestPerf}_{mdlNameRc}'; osMakedirs(pathBestPerf, exist_ok=True)
                    pathBstPerfPlot = f'{self.pathBstPerfPlot}_{mdlNameRc}'; osMakedirs(pathBstPerfPlot, exist_ok=True)
                    pathPerfPlot    = f'{self.pathPerfPlot}_{mdlNameRc}\\{foldID}'; osMakedirs(pathPerfPlot, exist_ok=True)
                else:
                    pathPrdct       = f'{self.pathModelPrdct}\\{foldID}'; osMakedirs(pathPrdct, exist_ok=True)
                    pathFldSngle    = f'{self.pathPerfSngle}\\{foldID}'; osMakedirs(pathFldSngle, exist_ok=True) 
                    pathPerfIndiv   = f'{self.pathModel}\\{foldID}'; osMakedirs(pathPerfIndiv, exist_ok=True)
                    pathMdl         = f'{self.pathMdl}'
                    pathModelPrdct  = f'{self.pathModelPrdct}'
                    pathBestPerf    = f'{self.pathBestPerf}'
                    pathBstPerfPlot = f'{self.pathBstPerfPlot}'
                    pathPerfPlot    = f'{self.pathPerfPlot}'
                # if
            
                # Export representation layer
                self.reprLyr[foldID].save(f"{pathMdl}\\reprModel_{self.model_name}_{foldID}.keras")

                # Save the best architecture and hyperparameters
                self.Model[foldID].save(f"{pathMdl}\\model_{self.model_name}_{foldID}.keras")
                save_hyperpmtr(self.ID, self.PrmtrBest[foldID].values, foldID, self.bandToUse_ui, 
                               self.model_name)

                # Plot of Accuracy and Loss
                history_dict = self.history[foldID].history

                # Fit Keys
                if self.model_name == 'DCNN':
                    acc      = history_dict['accuracy']
                    val_acc  = history_dict['val_accuracy']
                    loss     = history_dict['loss']
                    val_loss = history_dict['val_loss'] 

                    # Plot Error-Epoch
                    epoch_range = range(1, len(acc)+1)
                    epoch_err   = self.plot_TrnVal(epoch_range, loss, val_loss, foldID, 'Loss')
                    epoch_acc   = self.plot_TrnVal(epoch_range, acc, val_acc, foldID, 'Accuracy')
                # if

                if useFGRC:
                    if isBinary_ui:
                        self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreIndvdl'] = DataFrame({'AUC': Series([self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['auc']]),
                                                                                                'F1': Series([self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['f1']]),
                                                                                                'Accuracy': Series([self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['acc']]),
                                                                                                'Recall': Series([self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['rec']]),
                                                                                                'Specificity':Series([self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['spe']]),
                                                                                                'Precision': Series([self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['pre']])})
                
                        self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreOvrall'] = DataFrame({'MattCorr': Series([self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['matC']]),
                                                                                                'Kappa': Series([self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['kap']]),
                                                                                                'MicroF1': Series([self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['micF']]),
                                                                                                'BalancedAcc': Series([self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['balAcc']]),
                                                                                                'MicroAUC': Series([self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['micAUC']]),
                                                                                                'MicGeoMean': Series([self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['micGeoMean']]),
                                                                                                'DOR': Series([self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['dor']]),
                                                                                                'AdjustedF': Series([self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['adjF']])})
                    else:
                        self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreIndvdl'] = DataFrame({'AUC': self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['auc'],
                                                                                                'F1': self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['f1'],
                                                                                                'Accuracy': self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['acc'],
                                                                                                'Recall': self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['rec'],
                                                                                                'Specificity':self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['spe'],
                                                                                                'Precision': self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['pre']},
                                                                                                index=['min','median','max'])
                
                        self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreOvrall'] = DataFrame({'MattCorr': self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['matC'],
                                                                                                'Kappa': self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['kap'],
                                                                                                'MicroF1': self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['micF'],
                                                                                                'BalancedAcc': self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['balAcc'],
                                                                                                'MicroAUC': self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['micAUC'],
                                                                                                'MicGeoMean': self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['micGeoMean'],
                                                                                                'DOR': self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['dor'],
                                                                                                'AdjustedF': self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['adjF']},
                                                                                                index=['min','meidan','max'])
                    # if
                
                    if not foldID == 'SF':
                        # Exporting scores
                        for namePerfScore in perMeasureLst_ui:
                            self.allFold_scores[namePerfScore][i_fold] = self.PrdctPrefFGRC['Tst'][foldID]['perfScoreLst'][namePerfScore]
                        # for
                    # if

                    try:
                        prdctScoreDf = DataFrame(self.PrdctPrefFGRC[nameDataSet][foldID]['prdctScore']); cLabel_prdctDf = DataFrame(self.PrdctPrefFGRC[nameDataSet][foldID]['cLabel'])
                        # Export the performance scores
                        with ExcelWriter(f'{pathPrdct}\\{classPrdct_gl}_{self.model_name}_{nameDataSet}_{foldID}.xlsx') as assmnt1Wrtr:
                            cLabel_prdctDf.to_excel(assmnt1Wrtr, sheet_name=f'{self.model_name}')

                        with ExcelWriter(f'{pathPrdct}\\{scorePrdct_gl}_{self.model_name}_{nameDataSet}_{foldID}.xlsx') as assmnt2Wrtr:
                            prdctScoreDf.to_excel(assmnt2Wrtr, sheet_name=f'{self.model_name}')
                        #
                    except Exception as e:
                        print(f'Error in exporting excel sheets for class labels & predictions for {self.model_name}:\n\t{e}',flush=True)
                    # try
                    
                    try:
                        # Export the performance scores
                        with ExcelWriter(f'{pathFldSngle}\\{scoreIndvdl_gl}_{self.model_name}_{nameDataSet}_{foldID}.xlsx') as assmnt1Wrtr:
                            self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreIndvdl'].to_excel(assmnt1Wrtr, sheet_name=f'{self.model_name}')

                        with ExcelWriter(f'{pathFldSngle}\\{scoreOvrall_gl}_{self.model_name}_{nameDataSet}_{foldID}.xlsx') as assmnt2Wrtr:
                            self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreOvrall'].to_excel(assmnt2Wrtr, sheet_name=f'{self.model_name}')
                        #
                    except Exception as e:
                        print(f'Error in exporting excel sheets for perfScoreIndvdl & perfScoreOvrall in {foldID} for {self.model_name}:\n\t{e}',flush=True)
                    # try

                    # Contingency Table
                    Plot_assmnt1 = self.plot(self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreIndvdl'], foldID, nameDataSet, 'Indvdl',
                                             pathPerfPlot)
                    Plot_assmnt2 = self.plot(self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreOvrall'], foldID, nameDataSet, 'Ovrall',
                                             pathPerfPlot)
                # if
            # for foldID
        # for nameDataSet
        
        try:
            with open(f'{pathModelPrdct}\\PrdctPerf_{self.model_name}.pckl','wb') as pf:  
                dump(self.PrdctPrefFGRC, pf)
        except Exception as e:
            print(f'Error in exporting pickle files for PrdctPerf for {self.model_name}:\n\t{e}',flush=True)
        # try

        # Export the Dataframe for Best
        if foldID != 'SF':
            if isBinary_ui:
                self.saveBest['indivdl'] = DataFrame({'AUC': Series([self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['auc']]),
                                                      'F1': Series([self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['f1']]),
                                                      'Accuracy': Series([self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['acc']]),
                                                      'Recall': Series([self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['rec']]),
                                                      'Specificity':Series([self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['spe']]),
                                                      'Precision': Series([self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['pre']])})
                
                self.saveBest['overall'] = DataFrame({'MattCorr': Series([self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['matC']]),
                                                      'Kappa': Series([self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['kap']]),
                                                      'MicroF1': Series([self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['micF']]),
                                                      'BalancedAcc': Series([self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['balAcc']]),
                                                      'MicroAUC': Series([self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['micAUC']]),
                                                      'MicGeoMean': Series([self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['micGeoMean']]),
                                                      'DOR': Series([self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['dor']]),
                                                      'AdjustedF': Series([self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['adjF']])})
            else:
                self.saveBest['indivdl'] = DataFrame({'AUC': self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['auc'],
                                                      'F1': self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['f1'],
                                                      'Accuracy': self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['acc'],
                                                      'Recall': self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['rec'],
                                                      'Specificity':self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['spe'],
                                                      'Precision': self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['pre']},index=['min','meidan','max'])
                
                self.saveBest['overall'] = DataFrame({'MattCorr': self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['matC'],
                                                      'Kappa': self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['kap'],
                                                      'MicroF1': self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['micF'],
                                                      'BalancedAcc': self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['balAcc'],
                                                      'MicroAUC': self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['micAUC'],
                                                      'MicGeoMean': self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['micGeoMean'],
                                                      'DOR': self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['dor'],
                                                      'AdjustedF': self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['adjF']},index=['min','meidan','max'])
            # if

            try:
                # Export the performance scores
                osMakedirs(pathBestPerf, exist_ok=True)
                with ExcelWriter(f'{pathBestPerf}\\{scoreIndvdl_gl}_{self.model_name}BestIndvdl.xlsx') as assmnt1Wrtr:
                    self.saveBest['indivdl'].to_excel(assmnt1Wrtr, sheet_name=f'{self.model_name}')

                with ExcelWriter(f'{pathBestPerf}\\{scoreOvrall_gl}_{self.model_name}BestOverall.xlsx') as assmnt2Wrtr:
                    self.saveBest['overall'].to_excel(assmnt2Wrtr, sheet_name=f'{self.model_name}')
                #
            except Exception as e:
                print(f'Error in exporting excel sheets for saveBestIndvdl & saveBestOverall for {self.model_name}:\n\t{e}',flush=True)
            # try
                
            # Confusion Matrix
            osMakedirs(pathBstPerfPlot, exist_ok=True)
            ConfusionMatrixDisplay(confusion_matrix=self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['confMtrx'], 
                                    display_labels=self.sbComb).plot()
            plt.title(f'Confusion Matrix: Best')
            plt.savefig(f"{pathBstPerfPlot}\\{foldID}BestCmtrx.png") 
            plt.close()

            allFoldScores = {'AUC':array(list(self.allFold_scores['auc'].values())),'F1':array(list(self.allFold_scores['f1'].values())),'Accuracy':array(list(self.allFold_scores['acc'].values())),
                            'Recall':array(list(self.allFold_scores['rec'].values())),'Specificity':array(list(self.allFold_scores['spe'].values())),'Precision':array(list(self.allFold_scores['pre'].values())),
                            'confMtrx':array(list(self.allFold_scores['confMtrx'].values())),'MattCorr':array(list(self.allFold_scores['matC'].values())),'Kappa':array(list(self.allFold_scores['kap'].values())),
                            'MicroAUC':array(list(self.allFold_scores['micAUC'].values())),'BalancedAcc':array(list(self.allFold_scores['balAcc'].values())),'MicroF1':array(list(self.allFold_scores['micF'].values())),
                            'micGeoMean':array(list(self.allFold_scores['micGeoMean'].values())),'DOR':array(list(self.allFold_scores['dor'].values())),'AdjustedF':array(list(self.allFold_scores['adjF'].values()))}
            try:
                with open(f'{pathBestPerf}\\Foldscores.pckl','wb') as savebest_pf:
                    dump(allFoldScores, savebest_pf)
                # with
            except Exception as e:
                print(f'Error in exporting pickle file for AllFoldScores.pckl: \n\t{e}', flush=True)
            # try
        # if
    #

    def hpTuner(self, build_model, fTrain, cTrain, fVal, cVal, fTst, foldID=None):
        '''
        Applies Keras Tuner

        Input:
        -----
        fTrain: training samples (2D) [#soi, #num_chs*num_feats]
        cTrain: training samples of class labels (1D)
        fVal  : validation samples (2D) [#soi, #num_chs*num_feats]
        cVal  : validation samples of class labels (1D)
        foldID: fold number
        '''
        # Paths
        if foldID != None:
            foldID = f'F{foldID}'
        else:
            foldID = 'SF'
        # if
        self.pathMdl = f'{self.pathBestModel}'; osMakedirs(self.pathMdl, exist_ok=True)
        pathLog      = f'{self.pathHptLogs}'; osMakedirs(pathLog, exist_ok=True)

        # Initiate the model
        tuner = customTuner(hypermodel=build_model, objective=ktObjective_gl, max_trials = self.max_trial, executions_per_trial=self.exetnPerTrial, directory=f'{pathLog}',
                            project_name=f'{folder_prjct_gl}_{self.model_name}_{foldID}', overwrite=True, num_initial_points=self.num_init_pt)
        
        # Search
        if self.model_name == 'AE':
            tuner.search(fTrain, fTrain, validation_data=(fVal,fVal))
        else:
            tuner.search(fTrain, cTrain, validation_data=(fVal,cVal))
        # if

        print(f"Finding the best {self.model_name} parameters ends....", flush=True)

        # Get best model
        self.Model[foldID] = tuner.get_best_models(num_models=numBestModels_gl)[0]
        
        # Get the best hyperparameters
        self.PrmtrBest[foldID] = tuner.get_best_hyperparameters()[0]
        print(self.PrmtrBest[foldID].values, flush=True)

        # To get the representation layer
        idxreprLyr           = NUM_LAYRS_GL*self.PrmtrBest[foldID]['cnn_layers']
        self.reprLyr[foldID] = Model(inputs=self.Model[foldID].input,outputs=self.Model[foldID].layers[idxreprLyr].output)

        # Get the training, validation set from the representation layer
        self.fTrnRepr = self.reprLyr[foldID].predict(fTrain)
        self.fValRepr = self.reprLyr[foldID].predict(fVal)
        self.fTstRepr = self.reprLyr[foldID].predict(fTst)

        # Parameters
        batchSize_best = self.PrmtrBest[foldID].values['batch_size']
        epochs_best    = self.PrmtrBest[foldID].values['epochs']

        # Fit
        if self.model_name == 'AE':
            self.history[foldID] = self.Model[foldID].fit(fTrain, fTrain, batch_size=batchSize_best, epochs=epochs_best, 
                                                          validation_data=(fVal, fVal), verbose=self.verbose)
        else:
            self.history[foldID] = self.Model[foldID].fit(fTrain, cTrain, batch_size=batchSize_best, epochs=epochs_best, 
                                                          validation_data=(fVal, cVal), verbose=self.verbose)
        # if
    #

    def hpTuner_kFold(self, build_model, num_fold, size_train, size_test, fTst, 
                      cTst, mdlNameRc, mdlClsRc):
        '''
        Applies StratifiedShuffleSplit as a cross validation method for kfold

        Input:
        -----
        num_fold  : number of kfold
        size_train: size of training samples
        size_test : size of testing samples
        '''
        # Get the index of the rsltQuery
        rsltQueryLkUp = self.rsltQuery.reset_index()

        sss = StratifiedShuffleSplit(n_splits=num_fold, train_size=size_train, test_size=size_test, random_state=1)

        for i_fold, (trainID, valID) in enumerate(sss.split(self.fTrainOrg, self.cTrainOrg)): # each fold
            foldID             = f'F{i_fold}'
            fTrn_fld, fVal_fld = self.fTrainOrg[trainID,:], self.fTrainOrg[valID,:]
            cTrn_fld, cVal_fld = self.cTrainOrg.to_numpy()[trainID], self.cTrainOrg.to_numpy()[valID]

            # Get the LookUp Table
            rsltQueryLkUp.loc[trainID,'ID_names'] = 'Trn'
            rsltQueryLkUp.loc[valID,'ID_names']   = 'Val'

            # Applying the hpTuner
            self.hpTuner(build_model, fTrn_fld, cTrn_fld, fVal_fld, cVal_fld, fTst, str(i_fold))

            self.ModelFGRC[f'{self.model_name}_{mdlNameRc}'] = mdlClsRc(self.ID, fTrn_fld, cTrn_fld, fVal_fld, cVal_fld, fTst, cTst, self.bandToUse_ui, num_fold_ui,
                                                                        self.size_train, self.size_test,self.rsltQuery, self.fTrainOrg, self.cTrainOrg, True, self.model_name, 
                                                                        mdlNameRc, self.reprLyr[foldID], self.sbComb,runAll='SF', foldIDLst_ui=self.foldIDLst_ui, foldID=str(i_fold))
            
            # Save parameters
            self.PrmtrRC[foldID] = self.ModelFGRC[f'{self.model_name}_{mdlNameRc}'].PrmtrBest[foldID]

            self.PrdctPrefFGRC['Trn'][foldID] = self.ModelFGRC[f'{self.model_name}_{mdlNameRc}'].PrdctPerf['Trn'][foldID]
            self.PrdctPrefFGRC['Val'][foldID] = self.ModelFGRC[f'{self.model_name}_{mdlNameRc}'].PrdctPerf['Val'][foldID]
            self.PrdctPrefFGRC['Tst'][foldID] = self.ModelFGRC[f'{self.model_name}_{mdlNameRc}'].PrdctPerf['Tst'][foldID]

            # Best score for each model from Validation
            self.bstScore[i_fold] = accuracy_score(cVal_fld, self.ModelFGRC[f'{self.model_name}_{mdlNameRc}'].PrdctPerf['Val'][foldID]['cLabel'])

            # Save the LookUp
            try:
                osMakedirs(self.pathModel + f'_{mdlNameRc}_{foldID}', exist_ok=True)
                with open(f'{self.pathModel}_{mdlNameRc}_{foldID}\\TrainVal_LkUp_{self.model_name}_{foldID}.pckl','wb') as pf1:
                    dump(rsltQueryLkUp, pf1)
                #
            except Exception as e:
                print(f'Error in exporting TrainVal_LkUp_{self.model_name}_{foldID}.pckl:\n\t{e}')
            # try
        # for

        # Best score for each model from Validation
        self.bstFGIndx = int(argmax(array(list(self.bstScore.values()))))

        # Get best parameters
        self.bstPrmtrFldRC = self.PrmtrRC[f'F{str(self.bstFGIndx)}']
        self.bstPrmtrFldFG = self.PrmtrBest[f'F{str(self.bstFGIndx)}']

        bestScrs = {'single':{'AUC':float(self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['auc']),
                              'F1':float(self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['f1']),
                              'Accuracy':float(self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['acc']),
                              'Recall':float(self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['rec']),
                              'Specificity':float(self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['spe']),
                              'Precision':float(self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['pre']),
                              'DOR':float(self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['dor']),
                              'AdjustedF':float(self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['adjF'])},      
                    'overall':{'confMtrx':self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['confMtrx'],
                               'MattCorr':float(self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['matC']),
                               'Kappa':float(self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['kap']),
                               'MicroAUC':float(self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['micAUC']),
                               'BalancedAcc':float(self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['balAcc']),
                               'MicroF1':float(self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['micF']),
                               'MicGeoMean':float(self.PrdctPrefFGRC['Tst'][f'F{str(self.bstFGIndx)}']['perfScoreLst']['micGeoMean'])}}
        try:
            osMakedirs(self.pathBestPerf + f'_{mdlNameRc}',exist_ok=True)
            with open(f'{self.pathBestPerf}_{mdlNameRc}\\BestScores.pckl','wb') as bestpf:
                dump(bestScrs, bestpf)
            # with
        except Exception as e:
            print(f'Error in exporting BestScores.pckl:\n\t{e}')
        # try
    #

    def plot_TrnVal(self, epoch_range, epchTrn, epchVal, foldID=None, title=str):
        '''
        Plots the training and validation curve. (ie loss or accuracy)

        Input:
        -----
        epoch_range: number of epochs
        epchTrn    : name in str ('loss')
        epchVal    : name in str ('val_loss')
        foldID   : fold number
        title      : name of the title (loss or accuracy)
        '''
        # Path to save the Error and Accuracy curve
        if foldID != None:
            pathErrorPlot = f'{self.pathModelPlot}_{foldID}'; osMakedirs(pathErrorPlot, exist_ok=True)
        else:
            pathErrorPlot = f'{self.pathModelPlot}'; osMakedirs(pathErrorPlot, exist_ok=True)
        # if

        plt.figure()
        plt.plot(epoch_range, epchTrn, 'r-', label=f'Training {title}')
        plt.plot(epoch_range, epchVal, 'b-', label=f'Validation {title}')
        plt.title(f'Training and Validation {title}')
        plt.xlabel('Epochs')
        plt.ylabel(title)
        plt.legend()
        plt.savefig(f"{pathErrorPlot}\\{self.model_name}-{title}_{foldID}.png")
        plt.close()
    #
    
    def plot(self, perfData, foldID, nameDataSet, name, path):
        '''
        Plots the performance scores using different visualization

        Input:
        -----
        perfData     : data used for visualization (2D)
        nameDataSet: name of the dataset
        foldID     : fold number
        '''
        # Path to save the Error and Accuracy curve
        if foldID != None:
            pathPerfPlot  = f'{path}'; osMakedirs(pathPerfPlot, exist_ok=True)
        else:
            pathPerfPlot  = path; osMakedirs(pathPerfPlot, exist_ok=True)
        # if

        # Confusion matrix
        ConfusionMatrixDisplay(confusion_matrix=self.PrdctPrefFGRC[nameDataSet][foldID]['perfScoreLst']['confMtrx'], 
                               display_labels=self.sbComb).plot()
        plt.title(f'Confusion Matrix: {nameDataSet}_statsFeatures')
        plt.savefig(f"{pathPerfPlot}\\{self.model_name}_{nameDataSet}_{foldID}_{name}_Cmtrx.png") 
        plt.close()

        # Bar-plots
        # Load the spreadsheet data
        dfPerf_data = perfData.reset_index().rename(columns={'index':'Stats'})
        dfPerf_data = dfPerf_data.melt(id_vars='Stats', var_name='Metric', value_name='Value')

        # Plotting the bar plot
        plt.figure(figsize=figSize_barplt_ui)
        barplot(data=dfPerf_data, x='Metric', y='Value', hue='Stats', palette='viridis')

        # Adding labels and title
        plt.title('Performance Metrics by Stat', fontsize=fontSize_title_ui)
        plt.xlabel('Metric', fontsize=fontSize_xy_ui)
        plt.ylabel('Value', fontsize=fontSize_xy_ui)
        plt.legend(title='Stat', fontsize=fontSize_lgnd_ui)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{pathPerfPlot}\\{self.model_name}_{nameDataSet}_{foldID}_{name}barplot.png") 
        plt.close()
    #
        
class dcnn(BaseML):
    actvtn_lst_dcnn       = Dcnn_ui['actvtn_lst_cnv2D_ui']
    drpout_lst_dcnn       = Dcnn_ui['drpout_lst_cnv2D_ui']
    exetns_per_trial_dcnn = Dcnn_ui['exetns_per_trial_cnv2D_ui']
    kernel_dcnn           = Dcnn_ui['kernel_cnv2D_ui']
    loss_dcnn             = Dcnn_ui['loss_cnv2D_ui'] 
    metric_compile_dcnn   = Dcnn_ui['metric_compile_cnv2D_ui']     # metric used for the deep learning models
    max_trials_dcnn       = Dcnn_ui['max_trials_cnv2D_ui']
    min_lyr_dcnn          = Dcnn_ui['min_lyr_cnv2D_ui']
    min_value_dcnn        = Dcnn_ui['min_value_cnv2D_ui']
    max_value_dcnn        = Dcnn_ui['max_value_cnv2D_ui']  
    optmzr_lst_dcnn       = Dcnn_ui['optmzr_lst_cnv2D_ui']
    pool_dcnn             = Dcnn_ui['pool_cnv2D_ui']
    step_dcnn             = Dcnn_ui['step_cnv2D_ui']  
    wght_lst_dcnn         = Dcnn_ui['wght_lst_cnv2D_ui']  

    def __init__(self, ID, fTrn, cTrn, fVal, cVal, fTst, cTst, bandToUse_ui, num_fold, size_train, size_test, 
                 rsltQuery, fTrain, cTrain, sbComb=tuple, mdlNameRc=None, mdlClsRc=None, runAll=str, foldIDLst_ui=list):  
        self.cTrn              = cTrn  
        self.cTrainOrg         = cTrain
        self.cTst              = cTst
        self.fVal              = fVal     
        self.cVal              = cVal     
        self.fTrn              = fTrn
        self.fTrainOrg         = fTrain
        self.fTst              = fTst
        self.model_name        = Dcnn_ui['model_name_ui']
        self.num_init_pts_dcnn = num_init_pts_cnv2D_ui[self.model_name]
        self.num_fold          = num_fold
        self.size_train        = size_train
        self.size_test         = size_test
        self.runOnce           = True
        if useViewModel_ui:
            self.ID       = f'View{ID}'
        else:
            self.ID       = f'{ID}'
        # if
        self.pathBestModel     = f'{pathOutput_ui}\\{self.ID}\\{bandToUse_ui}\\{folder_model_gl}\\{self.model_name}'
        self.pathHptLogs       = f'{pathOutput_ui}\\{self.ID}\\{bandToUse_ui}\\{folder_hptLog_gl}\\{self.model_name}'
        self.rsltQuery         = rsltQuery
        self.verbose           = Dcnn_ui['verbose_cnv2D_ui']

        super().__init__(ID, bandToUse_ui, self.model_name, dcnn.max_trials_dcnn, self.num_init_pts_dcnn, 
                         dcnn.exetns_per_trial_dcnn, sbComb, self.rsltQuery, foldIDLst_ui)

        if runAll == 'SF':
            # Single Fold
            self.hpTuner(self.buildModel,self.fTrn, self.cTrn, self.fVal, self.cVal, self.fTst)
        else:
            # Folds for Netowrk with RCs
            self.hpTuner_kFold(self.buildModel, self.num_fold, self.size_train, self.size_test, self.fTst, 
                               self.cTst, mdlNameRc, mdlClsRc)
        # if
        if mdlNameRc is not None:
            self.export(dataSetNameLst_gl, useFGRC=True, mdlNameRc=mdlNameRc)
        else:
            self.export(dataSetNameLst_gl, useFGRC=False, mdlNameRc=None)
        # if
    #

    def buildModel(self, hp):
        '''
        DCNN model
        '''

        # Hyerparamter used for tuning
        actvtn_hp = hp.Choice('activation', values=dcnn.actvtn_lst_dcnn)
        drpout_hp = hp.Choice('dropout', dcnn.drpout_lst_dcnn)    
        kernel_hp = hp.Choice('kernel', values=[dcnn.kernel_dcnn])
        optmzr_hp = hp.Choice('optimizer', dcnn.optmzr_lst_dcnn)    
        pool_hp   = hp.Choice('pooling', values=[dcnn.pool_dcnn])
        wght_hp   = hp.Choice('kernel_initializer',dcnn.wght_lst_dcnn)

        # Get the grid of X, Y and number for features
        sizeX = self.fTrn.shape[1]; sizeY = self.fTrn.shape[2]; num_frames = self.fTrn.shape[3]

        # Input shape
        inputTnsr = Input(shape=(sizeX, sizeY, num_frames))
        
        # Getting the highest number of layers
        maxNumLyr = int(floor(log10((sizeX/SIZE_LAST_LAYER_IMG_GL))/LOG_10_2_GL))

        model_ = inputTnsr

        # Number of layer
        for i_lyr in range(hp.Int("cnn_layers", min_value=dcnn.min_lyr_dcnn, max_value=maxNumLyr)):
            # Check
            if model_.shape[1] >= kernel_hp:
                model_ = Conv2D(filters=hp.Int(f'conv{i_lyr}', min_value=dcnn.min_value_dcnn, max_value=dcnn.max_value_dcnn, step=dcnn.step_dcnn), 
                                kernel_size=(kernel_hp,kernel_hp), kernel_initializer=wght_hp, activation=actvtn_hp, padding='same')(model_)
                model_ = MaxPooling2D(pool_size=(pool_hp,pool_hp), strides=(2,2), padding='same', name=f'pool_{i_lyr}')(model_)
            else:
                break
            # if
        # for
            
        # Flatten
        model_ = Flatten()(model_)

        # Dropout
        if hp.Boolean("dropout"):
            model_ = Dropout(rate=drpout_hp)(model_)
        # if

        # Last layer
        layerLast = Dense(num_classes_ui, activation="softmax")(model_)

        model = Model(inputs=inputTnsr, outputs=layerLast)

        # Model summary
        model.summary()

        # Compile
        model.compile(optimizer=optmzr_hp, loss=dcnn.loss_dcnn, metrics=[dcnn.metric_compile_dcnn])

        return model
    #
    
class ae(BaseML):
    actvtn_lst_ae       = Ae_ui['actvtn_lst_ae_ui']
    drpout_lst_ae       = Ae_ui['drpout_lst_ae_ui']
    exetns_per_trial_ae = Ae_ui['exetns_per_trial_ae_ui']
    kernel_ae           = Ae_ui['kernel_ae_ui']
    optmzr_lst_ae       = Ae_ui['optmzr_lst_ae_ui']
    pool_ae             = Ae_ui['pool_ae_ui']
    wght_lst_ae         = Ae_ui['wght_lst_ae_ui']
    min_lyr_ae          = Ae_ui['min_lyr_ae_ui']
    min_value_ae        = Ae_ui['min_value_ae_ui']
    max_value_ae        = Ae_ui['max_value_ae_ui']
    step_ae             = Ae_ui['step_ae_ui']
    max_trial_ae        = Ae_ui['max_trials_ae_ui']
    def __init__(self, ID, fTrn, cTrn, fVal, cVal, fTst, cTst, bandToUse_ui, num_fold, size_train, size_test, 
                 rsltQuery, fTrain, cTrain, sbComb=tuple, mdlNameRc=None, mdlClsRc=None, runAll=str, foldIDLst_ui=list):  
        self.cTrn            = cTrn  
        self.cTrainOrg       = cTrain     
        self.cVal            = cVal 
        self.cTst            = cTst
        self.fTrn            = fTrn
        self.fVal            = fVal 
        self.fTst            = fTst
        self.fTrainOrg       = fTrain    
        self.num_fold        = num_fold
        self.size_train      = size_train
        self.size_test       = size_test
        self.model_name      = Ae_ui['model_name_ui']
        self.num_init_pts_ae = num_init_pts_ae_ui[self.model_name]
        if useViewModel_ui:
            self.ID       = f'View{ID}'
        else:
            self.ID       = f'{ID}'
        # if
        self.pathBestModel   = f'{pathOutput_ui}\\{self.ID}\\{bandToUse_ui}\\{folder_model_gl}\\{self.model_name}'
        self.pathHptLogs     = f'{pathOutput_ui}\\{self.ID}\\{bandToUse_ui}\\{folder_hptLog_gl}\\{self.model_name}'
        self.rsltQuery       = rsltQuery
        self.verbose         = Ae_ui['verbose_ae_ui']

        super().__init__(ID, bandToUse_ui, self.model_name, ae.max_trial_ae, self.num_init_pts_ae, 
                         ae.exetns_per_trial_ae, sbComb, self.rsltQuery, foldIDLst_ui)

        if runAll == 'SF':
            # Single Fold
            self.hpTuner(self.buildModel, self.fTrn, self.cTrn, self.fVal, self.cVal, self.fTst)
        else:
            # Folds
            self.hpTuner_kFold(self.buildModel, self.num_fold, self.size_train, self.size_test, self.fTst,
                               self.cTst, mdlNameRc, mdlClsRc)
        # if
        if mdlNameRc is not None:
            self.export(dataSetNameLst_gl, useFGRC=True, mdlNameRc=mdlNameRc)
        else:
            self.export(dataSetNameLst_gl, useFGRC=False, mdlNameRc=None)
        # if
    #

    def buildModel(self, hp):
        '''
        AE model
        '''
        # Hyerparamter used for tuning
        actvtn_ae_hp = hp.Choice('activation', values=ae.actvtn_lst_ae)   
        optmzr_ae_hp = hp.Choice('optimizer', ae.optmzr_lst_ae)    
        wght_ae_hp   = hp.Choice('kernel_initializer',ae.wght_lst_ae)
        pool_ae_hp   = hp.Choice('pooling', values=[ae.pool_ae])
        kernel_ae_hp = hp.Choice('kernel', values=[ae.kernel_ae])

        # Get the grid of X, Y and number for features
        sizeX = self.fTrn.shape[1]; sizeY = self.fTrn.shape[2]; num_frames = self.fTrn.shape[3]

        # Input shape
        inputTnsr = Input(shape=(sizeX, sizeY, num_frames))

        # Getting the highest number of layers
        maxNumLyr = int(floor(log10((sizeX/SIZE_LAST_LAYER_IMG_GL))/LOG_10_2_GL))

        model_ = inputTnsr; encd_lyrs = dict()

        # Encoder
        for i_lyr in range(hp.Int("cnn_layers",min_value=ae.min_lyr_ae, max_value=maxNumLyr)):
            if model_.shape[1] >= kernel_ae_hp:
                filters = hp.Int(f'conv_{i_lyr}', min_value=ae.min_value_ae, max_value=ae.max_value_ae, step=ae.step_ae)
                model_  = Conv2D(filters=filters, kernel_size=(kernel_ae_hp,kernel_ae_hp), kernel_initializer=wght_ae_hp,
                                 activation=actvtn_ae_hp, padding='same')(model_)
                model_  = MaxPooling2D(pool_size=(pool_ae_hp,pool_ae_hp),padding='same',name=f'pool1_{i_lyr}')(model_)

                # encd_lyrs = ((filters, hp_wght, hp_actvtn))
                encd_lyrs[i_lyr] = (filters, wght_ae_hp, actvtn_ae_hp)
            else:
                break
            #
        # for

        # Decoder
        for j_lyr in range(len(encd_lyrs)-1,-1,-1):
            model_ = Conv2D(filters=encd_lyrs[j_lyr][0], kernel_size=(kernel_ae_hp,kernel_ae_hp), kernel_initializer=encd_lyrs[j_lyr][1],
                            activation=encd_lyrs[j_lyr][2], padding='same')(model_)
            model_ = UpSampling2D(size=(pool_ae_hp,pool_ae_hp))(model_)
        # for

        model_decoder = Conv2D(filters=num_frames,kernel_size=(kernel_ae_hp,kernel_ae_hp),kernel_initializer=encd_lyrs[j_lyr][1],
                               activation=encd_lyrs[j_lyr][2], padding='same')(model_)
        
        autoencoder = Model(inputs=inputTnsr, outputs=model_decoder)

        del encd_lyrs

        autoencoder.compile(optimizer=optmzr_ae_hp, loss='mean_squared_error')

        return autoencoder
    #




