module_name = 'Regular Classifiers'

'''
Version: v1.1.0

Description:
    CNN

Authors:
    Dr. Omer Muhammet Soysal
    Iphy Kelvin

Date Created     : 02/28/2025
Date Last Updated: 02/28/2025

Doc:
    <***>

Notes:
    <***>
'''

# CUSTOM IMPORTS
from config        import ConfigRc as CfgRc, ConfigStatF as CfgSF, ConfigFeatGen_HC as CfgFeatGen, ConfigMain as CfgMain
from network_tuner import customTuner
from ui            import All_ui, Rc_ui, Ann_ui, num_init_pts_rc_ui, Knn_ui, Rf_ui, Svm_ui, num_init_pts_fgrc_ui, Fgrc_ui, LoadPrepare_ui as LdPrep_ui
from util          import save_hyperpmtr, find_min_median_max, ratioCM

# OTHER IMPORTS
from imblearn.metrics        import specificity_score, geometric_mean_score
from keras_tuner.tuners      import SklearnTuner
from keras_tuner.oracles     import BayesianOptimizationOracle
from keras                   import Model
from keras.layers            import Dense, Flatten, Dropout, Input
import matplotlib.pyplot     as plt
from numpy                   import array, sqrt, argmax
from os                      import makedirs as osMakedirs
from pickle                  import dump
from pandas                  import DataFrame, Series, ExcelWriter
from seaborn                 import boxplot, violinplot, barplot, set_theme
from sklearn.metrics         import confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, fbeta_score, matthews_corrcoef, cohen_kappa_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow              import constant_initializer

# USER INTERFACE
# foldIDLst_ui      = All_ui['foldIDLst_ui']
metric_compile_ui = Ann_ui['metric_compile_ui']
num_classes_ui    = All_ui['numSbjsCmd_ui']
isExport_ui       = All_ui['isExport_ui']
isBinary_ui       = Rc_ui['isBinary_ui']
pathOutput_ui     = All_ui['pathOutput_ui']
verbose_ui        = Ann_ui['verbose_cnv2D_ui']
useViewModel_ui   = LdPrep_ui['useViewModel_ui']
strategy_ui       = All_ui['strategy_ui']

# HPT FOR KNN
nghbrMin_ui    = Knn_ui['nghbrmin_ui']
nghbrMax_ui    = Knn_ui['nghbrmax_ui']
lfMin_ui       = Knn_ui['lfmin_ui']
lfMax_ui       = Knn_ui['lfmax_ui']
knnWgt_lst_ui  = Knn_ui['knnwgt_lst_ui']
knnMtrc_lst_ui = Knn_ui['knnmtrc_lst_ui']

# HPT FOR RF
rfMxDpthMin_ui  = Rf_ui['rfMxdpthMin_ui']   
rfMxDpthMax_ui  = Rf_ui['rfMxdpthMax_ui']   
rfMinSpltMin_ui = Rf_ui['rfMinSplitMin_ui'] 
rfMinSpltMax_ui = Rf_ui['rfMinSplitMax_ui'] 
rfCritLst_ui    = Rf_ui['rfCritLst_ui']     
rfEstMin_ui     = Rf_ui['rfEstMin_ui']
rfEstMax_ui     = Rf_ui['rfEstMax_ui']
rfEstStep_ui    = Rf_ui['rfEstStp_ui']

# HPT FOR SVM
svm_cLst_ui    = Svm_ui['svm_cLst']    
svm_gamaLst_ui = Svm_ui['svm_gamaLst'] 
svm_krnLst_ui  = Svm_ui['svm_krnlLst']

# HPT FOR ANN
min_value_ann_ui = Ann_ui['min_value_ann_ui']  
max_value_ann_ui = Ann_ui['max_value_ann_ui']  
step_ann_ui      = Ann_ui['step_ann_ui']       
min_lyr_ann_ui   = Ann_ui['min_lyr_ann_ui']   
max_lyr_ann_ui   = Ann_ui['max_lyr_ann_ui']   

# FOR ALL REGULAR CLASSIFIERS
perMeasureLst_ui = Rc_ui['perMeasurLst_ui']
rcScore_ui       = Rc_ui['scoring_ui']
# subj_names_ui    = All_ui['subjectsAll_ui']

# PLOTS
fontSize_title_ui = Rc_ui['fontSize_title_ui']
fontSize_xy_ui    = Rc_ui['fontSize_xy_ui']
fontSize_lgnd_ui  = Rc_ui['fontSize_lgnd_ui']
figSize_barplt_ui = Rc_ui['figSize_barplt_ui']
rw_subplt_ui      = Rc_ui['rw_subplt_ui']
col_subplt_ui     = Rc_ui['col_subplt_ui']

# CONSTANTS
folder_best_gl      = CfgMain['folder_BstperScores_gl']
folder_bestplot_gl  = CfgMain['folder_BstperPlot_gl']
col_x2_gl           = CfgFeatGen['col_x2_gl'] 
col_y_gl            = CfgFeatGen['col_y_gl'] 
classPrdct_gl       = CfgMain['classPrdt_gl']
folder_prjct_gl     = CfgMain['folder_prjct_gl']
folder_hptLog_gl    = CfgMain['folder_hptLog_gl']
folder_prdct_gl     = CfgMain['folder_prdct_gl']
folder_model_gl     = CfgMain['folder_model_gl']
folder_modelPlot_gl = CfgMain['folder_modelPlot_gl']
folder_perScores_gl = CfgMain['folder_perScores_gl']
folder_perPlot_gl   = CfgMain['folder_perPlot_gl']
numFeat_gl          = CfgSF['feature_gl']
keys_prdctPerf_gl   = CfgMain['keys_prdctPerf_gl']
numBestModels_gl    = CfgMain['numBestModels_gl']
dataSetNameLst_gl   = CfgMain['DataSetName_gl']
scoreIndvdl_gl      = CfgMain['scoreIndvdl_gl']
scoreOvrall_gl      = CfgMain['scoreOvrall_gl']
scorePrdct_gl       = CfgMain['scorePrdct_gl']

# INITIALIZATIONS
# history         = {}
# model           = {}
# PrdctPerf       = {}
# PrmtrBest       = {}

class BaseML():
    max_trials_rc  = Rc_ui['max_trials_ui']
    ktObjective_gl = CfgRc['ktObjective_gl']

    def __init__(self, ID, bandToUse_ui, model_name, rsltQuery, num_init_pts_rc, sbComb, isFGRC=None, fgRepr=None, foldIDLst_ui=list):
        self.allFold_scores  = {}
        self.bstScore        = {}
        self.bandToUse_ui    = bandToUse_ui
        self.fgRepr          = fgRepr
        self.history         = {}
        self.Model           = {}
        self.model_name      = model_name
        self.num_init_pts_rc = num_init_pts_rc
        self.isFGRC          = isFGRC
        self.PrdctPerf       = {}
        self.PrmtrBest       = {}
        self.rsltQuery       = rsltQuery
        self.saveBest        = {}
        self.sbComb          = sbComb
        self.foldIDLst_ui    = foldIDLst_ui
        if useViewModel_ui:
            self.ID       = f'View{ID}'
        else:
            self.ID       = f'{ID}'
        # if
        self.pathHptLog      = f'{pathOutput_ui}\\{self.ID}\\{self.bandToUse_ui}\\{folder_hptLog_gl}\\{self.model_name}'
        self.pathModelPrdct  = f'{pathOutput_ui}\\{self.ID}\\{self.bandToUse_ui}\\{folder_prdct_gl}\\{self.model_name}'
        self.pathModel       = f'{pathOutput_ui}\\{self.ID}\\{self.bandToUse_ui}\\{folder_model_gl}\\{self.model_name}'
        self.pathModelPlot   = f'{pathOutput_ui}\\{self.ID}\\{self.bandToUse_ui}\\{folder_modelPlot_gl}\\{self.model_name}'
        self.pathPerfSngle   = f'{pathOutput_ui}\\{self.ID}\\{self.bandToUse_ui}\\{folder_perScores_gl}\\{self.model_name}'
        self.pathPerfPlot    = f'{pathOutput_ui}\\{self.ID}\\{self.bandToUse_ui}\\{folder_perPlot_gl}\\{self.model_name}'
        self.pathBestPerf    = f'{pathOutput_ui}\\{self.ID}\\{self.bandToUse_ui}\\{folder_best_gl}\\{self.model_name}'
        self.pathBstPerfPlot = f'{pathOutput_ui}\\{self.ID}\\{self.bandToUse_ui}\\{folder_bestplot_gl}\\{self.model_name}'

        for nameDataSet in dataSetNameLst_gl:
            self.PrdctPerf[nameDataSet] = {}
            for foldID in self.foldIDLst_ui:
                self.PrdctPerf[nameDataSet][foldID] = {}
                for key in keys_prdctPerf_gl:
                    self.PrdctPerf[nameDataSet][foldID][key] = {}
                # for
            # for
        # for
        for namePerfScore in perMeasureLst_ui:
            self.allFold_scores[namePerfScore] = {}
        # for
    #

    def export(self, dataSetNameLst_gl):
        '''
        Exports variables to Excel and pickle files. 
        '''
        for nameDataSet in dataSetNameLst_gl:
            for i_fold, foldID in enumerate(self.foldIDLst_ui):

                # Path
                pathPrdct     = f'{self.pathModelPrdct}\\{foldID}'; osMakedirs(pathPrdct, exist_ok=True)
                pathPerfIndiv = f'{self.pathModel}\\{foldID}'; osMakedirs(pathPerfIndiv, exist_ok=True)
                pathFldSngle  = f'{self.pathPerfSngle}\\{foldID}'; osMakedirs(pathFldSngle, exist_ok=True)

                if not self.model_name.endswith('ANN'):
                    # Save model and parameters
                    with open(f'{self.pathMdl}\\model_{self.model_name}_{foldID}.pckl','wb') as save_mdl:
                        dump(self.Model[foldID], save_mdl)
                    # with
                    
                    # best parameters
                    save_hyperpmtr(self.ID, self.PrmtrBest[foldID].values, foldID, self.bandToUse_ui,
                                    self.model_name)
                else:
                    # Save the best architecture and hyperparameters
                    self.Model[foldID].save(f"{self.pathMdl}\\model_{self.model_name}_{foldID}.keras")
                    save_hyperpmtr(self.ID, self.PrmtrBest[foldID].values, foldID, self.bandToUse_ui, 
                                   self.model_name)

                    # Plot of Accuracy and Loss
                    history_dict = self.history[foldID].history

                    # Fit Keys
                    acc      = history_dict['accuracy']
                    val_acc  = history_dict['val_accuracy']
                    loss     = history_dict['loss']
                    val_loss = history_dict['val_loss'] 

                    # Plot Error-Epoch
                    epoch_range = range(1, len(acc)+1)
                    self.plotTrnVal(epoch_range, loss, val_loss, foldID, 'Loss')
                    self.plotTrnVal(epoch_range, acc, val_acc, foldID,'Accuracy')
                # if
                
                if isBinary_ui:
                    self.PrdctPerf[nameDataSet][foldID]['perfScoreIndvdl'] = DataFrame({'AUC': Series([self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['auc']]),
                                                                                        'F1': Series([self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['f1']]),
                                                                                        'Accuracy': Series([self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['acc']]),
                                                                                        'Recall': Series([self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['rec']]),
                                                                                        'Specificity':Series([self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['spe']]),
                                                                                        'Precision': Series([self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['pre']])})
            
                    self.PrdctPerf[nameDataSet][foldID]['perfScoreOvrall'] = DataFrame({'MattCorr': Series([self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['matC']]),
                                                                                        'Kappa': Series([self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['kap']]),
                                                                                        'MicroF1': Series([self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['micF']]),
                                                                                        'BalancedAcc': Series([self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['balAcc']]),
                                                                                        'MicroAUC': Series([self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['micAUC']]),
                                                                                        'MicGeoMean': Series([self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['micGeoMean']]),
                                                                                        'DOR': Series([self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['dor']]),
                                                                                        'AdjustedF': Series([self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['adjF']])})
                else:
                    self.PrdctPerf[nameDataSet][foldID]['perfScoreIndvdl'] = DataFrame({'AUC': self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['auc'],
                                                                                        'F1': self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['f1'],
                                                                                        'Accuracy': self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['acc'],
                                                                                        'Recall': self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['rec'],
                                                                                        'Specificity':self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['spe'],
                                                                                        'Precision': self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['pre']},
                                                                                        index=['min','median','max'])
            
                    self.PrdctPerf[nameDataSet][foldID]['perfScoreOvrall'] = DataFrame({'MattCorr': self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['matC'],
                                                                                        'Kappa': self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['kap'],
                                                                                        'MicroF1': self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['micF'],
                                                                                        'BalancedAcc': self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['balAcc'],
                                                                                        'MicroAUC': self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['micAUC'],
                                                                                        'MicGeoMean': self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['micGeoMean'],
                                                                                        'DOR': self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['dor'],
                                                                                        'AdjustedF': self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['adjF']},
                                                                                        index=['min','median','max'])
                # if
                if not foldID == 'SF':
                    # Exporting scores
                    for namePerfScore in perMeasureLst_ui:
                        self.allFold_scores[namePerfScore][i_fold] = self.PrdctPerf['Tst'][foldID]['perfScoreLst'][namePerfScore]
                    # for
                # if

                try:
                    prdctScoreDf = DataFrame(self.PrdctPerf[nameDataSet][foldID]['prdctScore']); cLabel_prdctDf = DataFrame(self.PrdctPerf[nameDataSet][foldID]['cLabel'])
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
                        self.PrdctPerf[nameDataSet][foldID]['perfScoreIndvdl'].to_excel(assmnt1Wrtr, sheet_name=f'{self.model_name}')

                    with ExcelWriter(f'{pathFldSngle}\\{scoreOvrall_gl}_{self.model_name}_{nameDataSet}_{foldID}.xlsx') as assmnt2Wrtr:
                        self.PrdctPerf[nameDataSet][foldID]['perfScoreOvrall'].to_excel(assmnt2Wrtr, sheet_name=f'{self.model_name}')
                    #
                except Exception as e:
                    print(f'Error in exporting excel sheets for perfScoreIndvdl & perfScoreOvrall in {foldID} for {self.model_name}:\n\t{e}',flush=True)
                # try

                # Contingency Table
                Plot_assmnt1 = self.plot(self.PrdctPerf[nameDataSet][foldID]['perfScoreIndvdl'], foldID, nameDataSet, 'Indvdl')
                Plot_assmnt2 = self.plot(self.PrdctPerf[nameDataSet][foldID]['perfScoreOvrall'], foldID, nameDataSet, 'Ovrall')
            # for
        # for
        try:
            with open(f'{self.pathModelPrdct}\\PrdctPerf_{self.model_name}.pckl','wb') as pf:  
                dump(self.PrdctPerf, pf)
        except Exception as e:
            print(f'Error in exporting pickle files for PrdctPerf for {self.model_name}:\n\t{e}',flush=True)
        # try

        # Export the Dataframe for Best
        if foldID != 'SF':
            if isBinary_ui:
                self.saveBest['indivdl'] = DataFrame({'AUC': Series([self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['auc']]),
                                                    'F1': Series([self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['f1']]),
                                                    'Accuracy': Series([self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['acc']]),
                                                    'Recall': Series([self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['rec']]),
                                                    'Specificity':Series([self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['spe']]),
                                                    'Precision': Series([self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['pre']])})
                
                self.saveBest['overall'] = DataFrame({'MattCorr': Series([self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['matC']]),
                                                    'Kappa': Series([self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['kap']]),
                                                    'MicroF1': Series([self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['micF']]),
                                                    'BalancedAcc': Series([self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['balAcc']]),
                                                    'MicroAUC': Series([self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['micAUC']]),
                                                    'MicGeoMean': Series([self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['micGeoMean']]),
                                                    'DOR': Series([self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['dor']]),
                                                    'AdjustedF': Series([self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['adjF']])})
            else:
                self.saveBest['indivdl'] = DataFrame({'AUC': self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['auc'],
                                                    'F1': self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['f1'],
                                                    'Accuracy': self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['acc'],
                                                    'Recall': self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['rec'],
                                                    'Specificity':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['spe'],
                                                    'Precision': self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['pre']},index=['min','meidan','max'])
                
                self.saveBest['overall'] = DataFrame({'MattCorr': self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['matC'],
                                                    'Kappa': self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['kap'],
                                                    'MicroF1': self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['micF'],
                                                    'BalancedAcc': self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['balAcc'],
                                                    'MicroAUC': self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['micAUC'],
                                                    'MicGeoMean': self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['micGeoMean'],
                                                    'DOR': self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['dor'],
                                                    'AdjustedF': self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['adjF']},index=['min','meidan','max'])
            # if
        
            try:
                # Export the performance scores
                osMakedirs(self.pathBestPerf, exist_ok=True)
                with ExcelWriter(f'{self.pathBestPerf}\\{scoreIndvdl_gl}_{self.model_name}BestIndvdl.xlsx') as assmnt1Wrtr:
                    self.saveBest['indivdl'].to_excel(assmnt1Wrtr, sheet_name=f'{self.model_name}')

                with ExcelWriter(f'{self.pathBestPerf}\\{scoreOvrall_gl}_{self.model_name}BestOverall.xlsx') as assmnt2Wrtr:
                    self.saveBest['overall'].to_excel(assmnt2Wrtr, sheet_name=f'{self.model_name}')
                #
            except Exception as e:
                print(f'Error in exporting excel sheets for saveBestIndvdl & saveBestOverall for {self.model_name}:\n\t{e}',flush=True)
            # try
                
            # Confusion Matrix
            osMakedirs(self.pathBstPerfPlot, exist_ok=True)
            ConfusionMatrixDisplay(confusion_matrix=self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['confMtrx'], 
                                    display_labels=self.sbComb).plot()
            plt.title(f'Confusion Matrix: Best')
            plt.savefig(f"{self.pathBstPerfPlot}\\{foldID}BestCmtrx.png") 
            plt.close()

            allFoldScores = {'AUC':array(list(self.allFold_scores['auc'].values())),'F1':array(list(self.allFold_scores['f1'].values())),'Accuracy':array(list(self.allFold_scores['acc'].values())),
                            'Recall':array(list(self.allFold_scores['rec'].values())),'Specificity':array(list(self.allFold_scores['spe'].values())),'Precision':array(list(self.allFold_scores['pre'].values())),
                            'confMtrx':array(list(self.allFold_scores['confMtrx'].values())),'MattCorr':array(list(self.allFold_scores['matC'].values())),'Kappa':array(list(self.allFold_scores['kap'].values())),
                            'MicroAUC':array(list(self.allFold_scores['micAUC'].values())),'BalancedAcc':array(list(self.allFold_scores['balAcc'].values())),'MicroF1':array(list(self.allFold_scores['micF'].values())),
                            'micGeoMean':array(list(self.allFold_scores['micGeoMean'].values())),'DOR':array(list(self.allFold_scores['dor'].values())),'AdjustedF':array(list(self.allFold_scores['adjF'].values()))}
            try:
                with open(f'{self.pathBestPerf}\\Foldscores.pckl','wb') as savebest_pf:
                    dump(allFoldScores, savebest_pf)
                # with
            except Exception as e:
                print(f'Error in exporting pickle file for AllFoldScores.pckl: \n\t{e}', flush=True)
            # try
        # if
    #

    def hpTuner(self, build_model, fTrn, cTrn, fVal, cVal,  fTst, cTst, foldID=None):
        '''
        Applies Sklearn Tuner

        Input:
        -----
        build_model: what model is used
        fTrn       : training samples (2D) [num_Soi, num_channels*num_feats]
        cTrn       : training samples of class labels (1D)
        fVal       : validation samples (2D) [num_Soi, num_channels*num_feats]
        cVal       : validation samples of class labels (1D)
        foldID     : fold number
        isExport   : used for exporting files
        '''

        #Paths
        if foldID != None:
            foldID = f'F{foldID}'
        else:
            foldID = 'SF'
        # if

        self.pathMdl = f'{self.pathModel}'; osMakedirs(self.pathMdl, exist_ok=True)
        pathLog      = f'{self.pathHptLog}'; osMakedirs(pathLog, exist_ok=True)

        with strategy_ui.scope():
            tuner = SklearnTuner(oracle=BayesianOptimizationOracle(objective=BaseML.ktObjective_gl, max_trials=BaseML.max_trials_rc, num_initial_points=self.num_init_pts_rc),
                                 hypermodel=build_model, scoring=rcScore_ui,overwrite=True, project_name=f'{folder_prjct_gl}_{self.model_name}_{foldID}',
                                 directory=f'{pathLog}')

            # search for best
            tuner.search(fTrn, cTrn)

            # Get best model and parameters
            self.Model[foldID]     = tuner.get_best_models(num_models=numBestModels_gl)[0]
            self.PrmtrBest[foldID] = tuner.get_best_hyperparameters()[0]

            # Predictions
            self.predict(self.Model[foldID],fTrn,cTrn,'Trn',foldID) 
            self.predict(self.Model[foldID],fVal,cVal,'Val',foldID) 
            self.predict(self.Model[foldID],fTst,cTst,'Tst',foldID) 
        # with
    #

    def hpTuner_kfold(self, build_model, num_fold, size_train, size_test, fTst, cTst):
        '''
        Applies StratifiedShuffleSplit as a cross validation method for kfold

        Input:
        -----
        build_model: what model is used
        num_fold   : number of kfold
        size_train : size of training samples
        size_test  : size of testing samples
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

            # Apply Representation layer
            # Shape: [#samples, #chs, #timesamples, #wdws]
            if self.isFGRC:
                featureTrnFG = self.fgRepr.predict(fTrn_fld)
                featureValFG = self.fgRepr.predict(fVal_fld)
                featureTstFG = self.fgRepr.predict(fTst)

                # Reshape them
                fTrnFld = featureTrnFG.reshape(featureTrnFG.shape[0],featureTrnFG.shape[1]*featureTrnFG.shape[2]*featureTrnFG.shape[3])
                fValFld = featureValFG.reshape(featureValFG.shape[0],featureValFG.shape[1]*featureValFG.shape[2]*featureValFG.shape[3])
                fTstFld = featureTstFG.reshape(featureTstFG.shape[0],featureTstFG.shape[1]*featureTstFG.shape[2]*featureTstFG.shape[3])
            else:
                # Reshape
                fTrnFld = fTrn_fld.reshape(fTrn_fld.shape[0],fTrn_fld.shape[1]*fTrn_fld.shape[2]*fTrn_fld.shape[3])
                fValFld = fVal_fld.reshape(fVal_fld.shape[0],fVal_fld.shape[1]*fVal_fld.shape[2]*fVal_fld.shape[3])
                fTstFld = fTst.reshape(fTst.shape[0],fTst.shape[1]*fTst.shape[2]*fTst.shape[3])
            # if
            
            # Applying the hpTuner
            self.hpTuner(build_model, fTrnFld, cTrn_fld, fValFld, cVal_fld, fTstFld, cTst, str(i_fold))

            # Best score for each model from Validation
            self.bstScore[i_fold] = accuracy_score(cVal_fld, self.PrdctPerf['Val'][foldID]['cLabel'])

            # Save the LookUp
            try:
                osMakedirs(self.pathModel, exist_ok=True)
                with open(f'{self.pathModel}\\TrainVal_LkUp_{self.model_name}_{foldID}.pckl','wb') as pf1:
                    dump(rsltQueryLkUp, pf1)
                #
            except Exception as e:
                print(f'Error in exporting TrainVal_LkUp_{self.model_name}_{foldID}.pckl:\n\t{e}')
            # try
        # for
        
        # Get the index for best
        self.bstIndx = int(argmax(array(list(self.bstScore.values()))))

        # Get best parameters
        self.bestPrmtrFldRC = self.PrmtrBest[f'F{str(self.bstIndx)}']
        
        if isBinary_ui:
            bestScrs = {'single':{'AUC':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['auc']),'F1':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['f1']),
                                'Accuracy':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['acc']),'Recall':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['rec']),
                                'Specificity':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['spe']),'Precision':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['pre']),
                                'DOR':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['dor']),'AdjustedF':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['adjF'])},      
                        'overall':{'confMtrx':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['confMtrx'],'MattCorr':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['matC']),
                                'Kappa':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['kap']),'MicroAUC':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['micAUC']),
                                'BalancedAcc':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['balAcc']),'MicroF1':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['micF']),
                                'MicGeoMean':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['micGeoMean'])}}
        else:
            bestScrs = {'single':{'AUC':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['auc'],'F1':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['f1'],
                                'Accuracy':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['acc'],'Recall':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['rec'],
                                'Specificity':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['spe'],'Precision':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['pre'],
                                'DOR':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['dor'],'AdjustedF':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['adjF']},      
                        'overall':{'confMtrx':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['confMtrx'],'MattCorr':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['matC'],
                                'Kappa':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['kap'],'MicroAUC':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['micAUC'],
                                'BalancedAcc':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['balAcc'],'MicroF1':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['micF'],
                                'MicGeoMean':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['micGeoMean']}}

        try:
            osMakedirs(self.pathBestPerf,exist_ok=True)
            with open(f'{self.pathBestPerf}\\BestScores.pckl','wb') as bestpf:
                dump(bestScrs, bestpf)
            # with
        except Exception as e:
            print(f'Error in exporting BestScores.pckl:\n\t{e}')
        # try
    #

    def predict(self, model, feat, cLabel, nameDataSet=str, foldID=None):
        '''
        Calculatess the prediction of the classification problem 
        using regular classifiers.

        Input:
        ------
        model      : model used
        feat       : training sets (2D)  [num_Soi, num_channels*num_feats] 
        cLabel     : testing sets (1D vector)
        foldID     : fold number
        isBinary   : whether binary or not
        nameDataSet: name of the dataset
        '''

        if self.model_name.endswith('ANN'):
            prdctScore   = model.predict(feat)
            cLabel_prdct = argmax(prdctScore,axis=1)
            if isBinary_ui: 
                yScore_AUC = prdctScore[:,1]
            else:
                yScore_AUC = prdctScore
            # if
        else:

            # Prediction Results
            cLabel_prdct = model.predict(feat)
            prdctScore   = model.predict_proba(feat)
            if isBinary_ui:
                # Get predict_proba
                yScore_AUC = prdctScore[:,1]
            else:
                yScore_AUC = prdctScore
            # if
        # if
        self.PrdctPerf[nameDataSet][foldID]['cLabel']     = cLabel_prdct 
        self.PrdctPerf[nameDataSet][foldID]['prdctScore'] = prdctScore

        # Performance Score
        # Classification
        if isBinary_ui:
            # Individual Scores
            average_p   = 'binary'
            num_class   = 1
            acc         = accuracy_score(cLabel, cLabel_prdct)
            auc         = roc_auc_score(cLabel,  yScore_AUC)
            confMtrx    = confusion_matrix(cLabel,  cLabel_prdct)
            fp,fn,tp,tn = confusion_matrix(cLabel,  cLabel_prdct).ravel()

            # Overall Scores
            micAUC    = roc_auc_score(cLabel, yScore_AUC, average='micro')
            mattCorr = array([matthews_corrcoef(cLabel,cLabel_prdct)]*num_class)[0]
            kappa    = array([cohen_kappa_score(cLabel,cLabel_prdct)]*num_class)[0]
            micF1    = array([f1_score(cLabel,cLabel_prdct, average='micro')]*num_class)[0]
            balAcc   = array([balanced_accuracy_score(cLabel,cLabel_prdct)]*num_class)[0]
            micGMean = array([geometric_mean_score(cLabel,cLabel_prdct, average='micro')]*num_class)[0]

        else: # (Multiclass)
            # Individual Scores
            average_p   = None
            num_class   = All_ui['numSbjsCmd_ui']
            acc         = array([accuracy_score(cLabel, cLabel_prdct)]*num_class)
            auc         = roc_auc_score(cLabel, yScore_AUC, average=average_p, multi_class='ovr')
            confMtrx    = confusion_matrix(cLabel,  cLabel_prdct)
            fp,fn,tp,tn = ratioCM(confMtrx)

            # Overall Scores
            micAUC   = array([roc_auc_score(cLabel,yScore_AUC, average='micro', multi_class='ovr')]*num_class)
            mattCorr = array([matthews_corrcoef(cLabel,cLabel_prdct)]*num_class)
            kappa    = array([cohen_kappa_score(cLabel,cLabel_prdct)]*num_class)
            micF1    = array([f1_score(cLabel,cLabel_prdct, average='micro')]*num_class)
            balAcc   = array([balanced_accuracy_score(cLabel,cLabel_prdct)]*num_class)
            micGMean = array([geometric_mean_score(cLabel,cLabel_prdct, average='micro')]*num_class)
        # if

        # Individual Scores
        f1          = f1_score(cLabel, cLabel_prdct, average=average_p)
        precsn      = precision_score(cLabel, cLabel_prdct, average=average_p)
        recal       = recall_score(cLabel, cLabel_prdct, average=average_p)
        specf       = specificity_score(cLabel, cLabel_prdct, average=average_p)
        fbeta       = fbeta_score(cLabel, cLabel_prdct,average=average_p,beta=2)
        npv         = tn/(fp+tn)                # Negative Predictive value (NPV)
        dor         = (tp*tn)/(fp*fn)           #diagnostic_oddratio
        invF        = (1.25*(npv*(tn/(fp+tn))))/(0.25*npv + (tn/(fp+tn)))  # Inverse F_0.5
        adjFm       = sqrt(fbeta*invF)

        if isBinary_ui:
            self.PrdctPerf[nameDataSet][foldID]['perfScoreLst'] = {'auc':auc,'f1':f1,'acc':acc,'rec':recal,'spe':specf,'pre':precsn,'dor':dor,'adjF':adjFm,
                                                                   'matC':mattCorr,'kap':kappa,'micF':micF1,'balAcc':balAcc,'micAUC':micAUC,
                                                                   'micGeoMean':micGMean,'confMtrx':confMtrx}
        else:
            scoreLst = (auc, f1, acc, recal, specf, precsn, dor, adjFm, mattCorr, kappa, micF1, balAcc, micAUC, micGMean)
            self.PrdctPerf[nameDataSet][foldID]['perfScoreLst'] = {key: find_min_median_max(value) for value, key in zip(scoreLst,perMeasureLst_ui[:-1])}
            self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['confMtrx'] = confMtrx
        # if
    #

    def plotTrnVal(self, epoch_range, epchTrn, epchVal, foldID=None, title=str):
        '''
        Plots the training and validation curve. (ie loss or accuracy)

        Input:
        -----
        epoch_range: number of epochs
        epchTrn    : name in str ('loss')
        epchVal    : name in str ('val_loss')
        foldID    : fold number
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

    def plot(self, perfData, foldID, nameDataSet, name):
        '''
        Plots the performance scores using different visualization

        Input:
        -----
        perfData   : data used for visualization (2D)
        nameDataSet: name of the dataset
        foldID     : fold number
        nameDataSet: Trn, Val or Tst
        name       : Ovrall or Indvdl
        '''
        # Path to save the Error and Accuracy curve
        if foldID != None:
            pathPerfPlot  = f'{self.pathPerfPlot}_{foldID}'; osMakedirs(pathPerfPlot, exist_ok=True)
        else:
            pathPerfPlot  = self.pathPerfPlot; osMakedirs(pathPerfPlot, exist_ok=True)
        # if

        # Confusion matrix
        ConfusionMatrixDisplay(confusion_matrix=self.PrdctPerf[nameDataSet][foldID]['perfScoreLst']['confMtrx'], 
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


class ANN(BaseML):
    actvtn_lst_ann       = Ann_ui['actvtn_lst_ann_ui']
    bias_value_ann       = Ann_ui['bias_value_ann_ui']
    drpout_lst_ann       = Ann_ui['drpout_lst_ann_ui']
    exetns_per_trial_ann = Ann_ui['exetns_per_trial_ann_ui']
    ktObjective_ann      = CfgRc['ktObjectiveAnn_gl']
    max_trials_ann       = Ann_ui['max_trials_ann_ui']
    optimzr_lst_ann      = Ann_ui['optmzr_lst_ann_ui']
    wght_lst_ann         = Ann_ui['wght_lst_ann_ui'] 
    
    def __init__(self, ID, fTrn, cTrn, fVal, cVal, fTst, cTst, bandToUse_ui, num_fold, size_train, size_test, rsltQuery, fTrain, cTrain, 
                 isFGRC=None, mdlNameFG=None, mdlNameRc=None, fgRepr=None, sbComb=tuple, runAll=str, foldIDLst_ui=list, foldID=None):
        self.cTrn        = cTrn   
        self.cTrainOrg   = cTrain
        self.cVal        = cVal
        self.cTst        = cTst
        self.fTrn        = fTrn 
        self.fVal        = fVal 
        self.fTst        = fTst 
        self.fTrainOrg   = fTrain
        self.num_fold    = num_fold
        self.size_train  = size_train
        self.size_test   = size_test
        self.rsltQuery   = rsltQuery
        if isFGRC:
            self.model_name       = Fgrc_ui[f'model_name_{mdlNameFG}_{mdlNameRc}_ui']
            self.num_init_pts_ann = num_init_pts_fgrc_ui[f'{mdlNameFG}_{mdlNameRc}']
        else:
            self.model_name       = Ann_ui['model_name_ui']
            self.num_init_pts_ann = num_init_pts_rc_ui[self.model_name]
        # if
    
        super().__init__(ID, bandToUse_ui, self.model_name, self.rsltQuery, self.num_init_pts_ann, 
                         sbComb, isFGRC, fgRepr, foldIDLst_ui)

        if runAll == 'SF':
            # Single Fold
            self.hpTuner(self.fTrn, self.cTrn, self.fVal, self.cVal, self.fTst, self.cTst, foldID)
        else:
            # Single Fold
            self.hpTuner(self.fTrn, self.cTrn, self.fVal, self.cVal, self.fTst, self.cTst, foldID)

            # Folds
            self.hpTuner_kfold(self.num_fold, self.size_train, self.size_test, fTst, self.cTst)

            self.export(dataSetNameLst_gl)
        # if
    #

    def buildModel(self, hp):
        '''
        Uses ANN model
        '''

        # Hyperparameters
        hp_actvtn = hp.Choice('activation', values=ANN.actvtn_lst_ann)
        hp_drpout = hp.Choice('dropout', ANN.drpout_lst_ann)    
        hp_optmzr = hp.Choice('optimizer', ANN.optimzr_lst_ann)    
        hp_wght   = hp.Choice('kernel_initializer',ANN.wght_lst_ann)

        # Get the grid of X, Y and number for features
        sizeX = self.fTrn.shape[1]; sizeY = self.fTrn.shape[2]; num_frames = self.fTrn.shape[3]

        # Read the input shape
        inputImg  = Input(shape=(sizeX, sizeY, num_frames))
        outputImg = Flatten()(inputImg)

        model_ = outputImg
        
        # Generate model Layers
        for i in range(hp.Int("ann_layers", min_value=min_lyr_ann_ui, max_value=max_lyr_ann_ui)):
            model_ = Dense(units=hp.Int(f"units_{i}",min_value=min_value_ann_ui, max_value=max_value_ann_ui, step=step_ann_ui),
                           activation=hp_actvtn, kernel_initializer=hp_wght, bias_initializer=constant_initializer(ANN.bias_value_ann), 
                           name=f'Dense_{i}')(model_)
        # for

        # Apply Dropout or not
        if hp.Boolean("dropout"):
            model_ = Dropout(rate=hp_drpout)(model_)
        # if

        # Last layer
        last_lyr = Dense(num_classes_ui, activation="softmax")(model_)

        #end Generate Layers
        ntwrk = Model(inputs=inputImg, outputs=last_lyr)

        # Compile model
        ntwrk.compile(optimizer=hp_optmzr, loss='sparse_categorical_crossentropy',
                      metrics=[metric_compile_ui])
        
        return ntwrk
    #

    def hpTuner(self, fTrn, cTrn, fVal, cVal, fTst, cTst, foldID=None):
        '''
        Applies Keras Tuner

        Input:
        -----
        build_model: what model is used
        fTrn     : training samples (2D) [#soi, #num_chs*num_feats]
        cTrn     : training samples of class labels (1D)
        fVal       : validation samples (2D) [#soi, #num_chs*num_feats]
        cVal       : validation samples of class labels (1D)
        foldID   : fold number
        isExport   : used for exporting files
        '''

        #Paths
        if foldID != None:
            foldID  = f'F{foldID}'
        else:
            foldID = 'SF'
        # if
        self.pathMdl = f'{self.pathModel}'; osMakedirs(self.pathMdl, exist_ok=True)
        pathLog      = f'{self.pathHptLog}'; osMakedirs(pathLog, exist_ok=True)

        tuner = customTuner(hypermodel=self.buildModel,objective=ANN.ktObjective_ann, max_trials = ANN.max_trials_ann, 
                            executions_per_trial=ANN.exetns_per_trial_ann, directory=f'{pathLog}',
                            project_name=f'{folder_prjct_gl}_{self.model_name}', overwrite=True, 
                            num_initial_points=self.num_init_pts_ann)

        # search for best
        tuner.search(fTrn, cTrn, validation_data=(fVal, cVal))

        print(f"Finding the best {self.model_name} parameters ends....", flush=True)

        # Get best model and best hyperparameters
        self.Model[foldID]     = tuner.get_best_models(num_models=numBestModels_gl)[0]
        self.PrmtrBest[foldID] = tuner.get_best_hyperparameters()[0]

        # Parameters
        batchSize_best = self.PrmtrBest[foldID].values['batch_size']
        epochs_best    = self.PrmtrBest[foldID].values['epochs']

        # Fit
        self.history[foldID] = self.Model[foldID].fit(fTrn, cTrn, batch_size=batchSize_best, epochs=epochs_best, 
                                                      validation_data=(fVal, cVal), verbose=verbose_ui)
        
        # To get the predictions for training and validation
        self.predict(self.Model[foldID],fTrn,cTrn,'Trn', foldID)
        self.predict(self.Model[foldID],fVal,cVal,'Val', foldID)
        self.predict(self.Model[foldID],fTst,cTst,'Tst', foldID)
    #

    def hpTuner_kfold(self, num_fold, size_train, size_test, fTst, cTst):
        '''
        Applies StratifiedShuffleSplit as a cross validation method for kfold

        Input:
        -----
        num_fold   : number of kfold
        size_train : size of training samples
        size_test  : size of testing samples
        '''
        rsltQueryLkUp = self.rsltQuery.reset_index()
        
        sss = StratifiedShuffleSplit(n_splits=num_fold, train_size=size_train, test_size=size_test, random_state=1)

        for i_fold, (trainID, valID) in enumerate(sss.split(self.fTrainOrg, self.cTrainOrg)): # each fold
            foldID      = f'F{i_fold}'
            fTrn_fld, fVal_fld = self.fTrainOrg[trainID,:], self.fTrainOrg[valID,:]
            cTrn_fld, cVal_fld = self.cTrainOrg.to_numpy()[trainID], self.cTrainOrg.to_numpy()[valID]

            # Get the LookUp Table
            rsltQueryLkUp.loc[trainID,'ID_names'] = 'Trn'
            rsltQueryLkUp.loc[valID,'ID_names']   = 'Val'

            # Apply Representation layer
            if self.isFGRC:
                fTrn_fld = self.fgRepr.reprLyr.predict(fTrn_fld)
                fVal_fld = self.fgRepr.reprLyr.predict(fVal_fld)
                fTst_fld = self.fgRepr.reprLyr.predict(fTst)
            else:
                fTst_fld = fTst

            # Applying the hpTuner
            self.hpTuner(fTrn_fld, cTrn_fld, fVal_fld, cVal_fld, fTst_fld, cTst, str(i_fold))

            # Best score for each model from Validation
            self.bstScore[i_fold] = accuracy_score(cVal_fld, self.PrdctPerf['Val'][foldID]['cLabel'])

            # Save the LookUp
            try:
                osMakedirs(self.pathModel, exist_ok=True)
                with open(f'{self.pathModel}\\TrainVal_LkUp_{self.model_name}_{foldID}.pckl','wb') as pf1:
                    dump(rsltQueryLkUp, pf1)
                #
            except Exception as e:
                print(f'Error in exporting TrainVal_LkUp_{self.model_name}_{foldID}.pckl:\n\t{e}')
            # try
        # for

        # Get the index for best
        self.bstIndx = int(argmax(array(list(self.bstScore.values()))))

        # Get best parameters
        self.bestPrmtrFldRC = self.PrmtrBest[f'F{str(self.bstIndx)}']
        
        if isBinary_ui:
            bestScrs = {'single':{'AUC':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['auc']),'F1':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['f1']),
                                  'Accuracy':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['acc']),'Recall':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['rec']),
                                  'Specificity':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['spe']),'Precision':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['pre']),
                                  'DOR':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['dor']),'AdjustedF':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['adjF'])},      
                        'overall':{'confMtrx':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['confMtrx'],'MattCorr':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['matC']),
                                   'Kappa':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['kap']),'MicroAUC':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['micAUC']),
                                   'BalancedAcc':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['balAcc']),'MicroF1':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['micF']),
                                   'MicGeoMean':float(self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['micGeoMean'])}}
        else:
            bestScrs = {'single':{'AUC':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['auc'],'F1':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['f1'],
                                  'Accuracy':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['acc'],'Recall':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['rec'],
                                  'Specificity':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['spe'],'Precision':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['pre'],
                                  'DOR':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['dor'],'AdjustedF':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['adjF']},      
                        'overall':{'confMtrx':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['confMtrx'],'MattCorr':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['matC'],
                                   'Kappa':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['kap'],'MicroAUC':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['micAUC'],
                                   'BalancedAcc':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['balAcc'],'MicroF1':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['micF'],
                                   'MicGeoMean':self.PrdctPerf['Tst'][f'F{str(self.bstIndx)}']['perfScoreLst']['micGeoMean']}}

        try:
            osMakedirs(self.pathBestPerf,exist_ok=True)
            with open(f'{self.pathBestPerf}\\BestScores.pckl','wb') as bestpf:
                dump(bestScrs, bestpf)
            # with
        except Exception as e:
            print(f'Error in exporting BestScores.pckl:\n\t{e}')
        # try
    #

class KNN(BaseML):
    def __init__(self, ID, fTrn, cTrn, fVal, cVal, fTst, cTst, bandToUse_ui, num_fold, size_train, size_test, rsltQuery, fTrain, cTrain, 
                 isFGRC=None, mdlNameFG=None, mdlNameRc=None, fgRepr=None, sbComb=tuple, runAll=str, foldIDLst_ui=list, foldID=None):
        self.cVal        = cVal
        self.cTrainOrg   = cTrain
        self.cTrn        = cTrn
        self.cTst        = cTst
        self.fTrn        = fTrn.reshape(fTrn.shape[0],fTrn.shape[1]*fTrn.shape[2]*fTrn.shape[3])  
        self.fTrainOrg   = fTrain   
        self.fVal        = fVal.reshape(fVal.shape[0],fVal.shape[1]*fVal.shape[2]*fVal.shape[3])  
        self.fTst        = fTst.reshape(fTst.shape[0],fTst.shape[1]*fTst.shape[2]*fTst.shape[3]) 
        self.num_fold    = num_fold
        self.size_train  = size_train
        self.size_test   = size_test
        self.rsltQuery   = rsltQuery
        if isFGRC:
            self.model_name      = Fgrc_ui[f'model_name_{mdlNameFG}_{mdlNameRc}_ui'] 
            self.num_init_pts_rc = num_init_pts_fgrc_ui[f'{mdlNameFG}_{mdlNameRc}']
        else:
            self.model_name      = Knn_ui['model_name_ui']
            self.num_init_pts_rc = num_init_pts_rc_ui[self.model_name]
        # if
    
        super().__init__(ID, bandToUse_ui, self.model_name, self.rsltQuery, self.num_init_pts_rc, 
                         sbComb, isFGRC, fgRepr, foldIDLst_ui)
        
        if runAll == 'SF':
            # Single Fold
            self.hpTuner(self.buildModel, self.fTrn, self.cTrn, self.fVal, self.cVal, self.fTst, self.cTst, foldID)
        else:
            # Single Fold
            self.hpTuner(self.buildModel, self.fTrn, self.cTrn, self.fVal, self.cVal, self.fTst, self.cTst, foldID)

            # Folds
            self.hpTuner_kfold(self.buildModel, self.num_fold, self.size_train, self.size_test, fTst, self.cTst)

            self.export(dataSetNameLst_gl)
        # if
    #

    def buildModel(self, hp):
        '''
        Uses KNN model
        '''
        # Hyperparameters
        hp_neghbr = hp.Int('n_neighbors', nghbrMin_ui, nghbrMax_ui)
        hp_lfsize = hp.Int('leaf_size', lfMin_ui, lfMax_ui)
        hp_wghts  = hp.Choice('weights', knnWgt_lst_ui)
        hp_mtric  = hp.Choice('metric', knnMtrc_lst_ui)

        # Initiate the model
        model_knn = KNeighborsClassifier(n_neighbors=hp_neghbr, weights=hp_wghts, leaf_size=hp_lfsize,
                                         metric=hp_mtric)

        return model_knn
    #

class RF(BaseML):
    def __init__(self, ID, fTrn, cTrn, fVal, cVal,  fTst, cTst, bandToUse_ui, num_fold, size_train, size_test, rsltQuery, fTrain, cTrain, 
                 isFGRC=None, mdlNameFG=None, mdlNameRc=None, fgRepr=None, sbComb=tuple, runAll=str, foldIDLst_ui=list, foldID=None):
        self.cTrn        = cTrn
        self.cVal        = cVal
        self.cTrainOrg   = cTrain
        self.cTst        = cTst
        self.fTrn        = fTrn.reshape(fTrn.shape[0],fTrn.shape[1]*fTrn.shape[2]*fTrn.shape[3])     
        self.fVal        = fVal.reshape(fVal.shape[0],fVal.shape[1]*fVal.shape[2]*fVal.shape[3])   
        self.fTst        = fTst.reshape(fTst.shape[0],fTst.shape[1]*fTst.shape[2]*fTst.shape[3])  
        self.fTrainOrg   = fTrain
        self.num_fold    = num_fold
        self.size_train  = size_train
        self.size_test   = size_test
        self.rsltQuery   = rsltQuery
        
        if isFGRC:
            self.model_name      = Fgrc_ui[f'model_name_{mdlNameFG}_{mdlNameRc}_ui'] 
            self.num_init_pts_rc = num_init_pts_fgrc_ui[f'{mdlNameFG}_{mdlNameRc}']
        else:
            self.model_name      = Rf_ui['model_name_ui']
            self.num_init_pts_rc = num_init_pts_rc_ui[self.model_name]
        # if
    
        super().__init__(ID, bandToUse_ui, self.model_name, self.rsltQuery, self.num_init_pts_rc, 
                         sbComb, isFGRC, fgRepr, foldIDLst_ui)

        if runAll == 'SF':
            # Single Fold
            self.hpTuner(self.buildModel, self.fTrn, self.cTrn, self.fVal, self.cVal, self.fTst, self.cTst, foldID)
        else:
            # Single Fold
            self.hpTuner(self.buildModel, self.fTrn, self.cTrn, self.fVal, self.cVal, self.fTst, self.cTst, foldID)

            # Folds
            self.hpTuner_kfold(self.buildModel, self.num_fold, self.size_train, self.size_test, fTst, self.cTst)

            self.export(dataSetNameLst_gl)
        # if
    #

    def buildModel(self, hp):
        '''
        Uses RF model
        '''
        # Hyperparameters
        hp_mxdepth      = hp.Int('max_depth', rfMxDpthMin_ui, rfMxDpthMax_ui) 
        hp_minsmplsplit = hp.Int('min_samples_split', rfMinSpltMin_ui, rfMinSpltMax_ui)
        hp_critrn       = hp.Choice('criterion', rfCritLst_ui)
        hp_estmtr       = hp.Int('n_estimators', rfEstMin_ui, rfEstMax_ui, step=rfEstStep_ui)

        # Initiate the model
        model_rf = RandomForestClassifier(n_estimators=hp_estmtr, criterion=hp_critrn, max_features=len(numFeat_gl),
                                          max_depth=hp_mxdepth, min_samples_split=hp_minsmplsplit)

        return model_rf
    #

class SVM(BaseML):
    def __init__(self, ID, fTrn, cTrn, fVal, cVal,  fTst, cTst, bandToUse_ui, num_fold, size_train, size_test, rsltQuery, fTrain, cTrain, 
                 isFGRC=None, mdlNameFG=None, mdlNameRc=None, fgRepr=None, sbComb=tuple, runAll=str, foldIDLst_ui=list, foldID=None):
        self.cTrn        = cTrn
        self.cVal        = cVal
        self.cTrainOrg   = cTrain
        self.cTst        = cTst
        self.fTrn        = fTrn.reshape(fTrn.shape[0],fTrn.shape[1]*fTrn.shape[2]*fTrn.shape[3])  
        self.fVal        = fVal.reshape(fVal.shape[0],fVal.shape[1]*fVal.shape[2]*fVal.shape[3]) 
        self.fTst        = fTst.reshape(fTst.shape[0],fTst.shape[1]*fTst.shape[2]*fTst.shape[3])   
        self.fTrainOrg   = fTrain
        self.num_fold    = num_fold
        self.size_train  = size_train
        self.size_test   = size_test
        self.rsltQuery   = rsltQuery
        
        if isFGRC:
            self.model_name      = Fgrc_ui[f'model_name_{mdlNameFG}_{mdlNameRc}_ui'] 
            self.num_init_pts_rc = num_init_pts_fgrc_ui[f'{mdlNameFG}_{mdlNameRc}']
        else:
            self.model_name      = Svm_ui['model_name_ui']
            self.num_init_pts_rc = num_init_pts_rc_ui[self.model_name]
        # if
    
        super().__init__(ID, bandToUse_ui, self.model_name, self.rsltQuery, self.num_init_pts_rc, 
                         sbComb, isFGRC, fgRepr, foldIDLst_ui)

        if runAll == 'SF':
            # Single Fold
            self.hpTuner(self.buildModel, self.fTrn, self.cTrn, self.fVal, self.cVal, self.fTst, self.cTst, foldID)
        else:
            # Single Fold
            self.hpTuner(self.buildModel, self.fTrn, self.cTrn, self.fVal, self.cVal, self.fTst, self.cTst, foldID)

            # Folds
            self.hpTuner_kfold(self.buildModel, self.num_fold, self.size_train, self.size_test, fTst, self.cTst)

            self.export(dataSetNameLst_gl)
        # if
    #

    def buildModel(self, hp):
        '''
        Uses SVM model
        '''
        # Hyperparameters
        hp_c    = hp.Choice('C', svm_cLst_ui)
        hp_gmma = hp.Choice('gamma', svm_gamaLst_ui)
        hp_krnl = hp.Choice('kernel', svm_krnLst_ui)

        # Initiate the model
        model_svm = SVC(C=hp_c, kernel=hp_krnl, gamma=hp_gmma, probability=True)

        return model_svm
    #


            