module_name = 'Visualization'

'''
Version: v1.3.1

Description:
   Visualization

Authors:
    Dr. Omer Muhammet Soysal
    Iphy Kelvin

Date Created     : 10/18/2024
Date Last Updated: 10/18/2024

Doc:


Notes:
    <***>
'''


# CUSTOM IMPORTS
from config_compile import ConfigMainCmp, ConfigModelsCmp
from ui             import All_ui
from os             import makedirs as osMakedirs

# OTHER IMPORTS
from pandas              import ExcelWriter
import matplotlib.pyplot as plt
from seaborn             import violinplot, set_theme, barplot

# USER INTERFACE
pathOutput_ui = All_ui['pathOutput_ui']

# CONSTANTS
folderExprt_gl      = ConfigMainCmp['folderExprt_gl']
col_subplt_gl       = ConfigMainCmp['col_subplt_gl']
col_x1_gl           = ConfigMainCmp['col_x1_gl'] 
col_x2_gl           = ConfigMainCmp['col_x2_gl']       
col_y_gl            = ConfigMainCmp['col_y_gl']    
rw_subplt_gl        = ConfigMainCmp['rw_subplt_gl']
RepType_gl          = ConfigMainCmp['RepType_gl']
modelAllRep_gl      = ConfigModelsCmp['modelAllRep_gl']
modelNameDsplyRc_gl = ConfigModelsCmp['modelNameDsplyRc_gl']
nameStat_gl         = ConfigMainCmp['nameStat_gl'] 
modelAllRc_gl       = ConfigModelsCmp['modelAllRc_gl']
  


class Visualizer():
    def __init__(self, data, path, pmList, fldrName=None):
        self.data       = data
        self.pmList     = pmList     # [IPM, OPM]
        self.pathExport = path
        if fldrName != None:
            self.fldrName = fldrName   # Level_1
        # if
        self.pathViolin = f'{self.pathExport}\\{self.pmList[0]}'; osMakedirs(self.pathViolin, exist_ok=True)
        self.pathplots_bar = f'{self.pathExport}{self.pmList[0]}'; osMakedirs(self.pathplots_bar, exist_ok=True)

        self.barPlot()        
    #

    def barPlotSub(self,pathPlt,pm_name,nmSt):

        pathFig  = f'{pathPlt}\\Fig'; osMakedirs(pathFig, exist_ok=True)
        pathData = f'{pathPlt}\\Data'; osMakedirs(pathData, exist_ok=True)
        fName    = f'{nmSt}_{pm_name}'

        if 'ec' in pathPlt.split('\\')[2]:
            stmlsName = 'ec'.upper()
        elif 'eo' in pathPlt.split('\\')[2]:
            stmlsName = 'eo'.upper()
        # if
        sgnlCndtn = pathPlt.split('\\')[1][3:6]

        try:
            fig = plt.figure(figsize=(30, 18))
            barplot(data=self.data, x=self.data['RepType'], y=self.data['Score'], 
                    hue=self.data['Model'])
            plt.title(f'Comparison of ML Models across Representations ({stmlsName}, {sgnlCndtn})')
            plt.xlabel('Representation')
            plt.ylabel(f'{nmSt} {pm_name} Score')
            plt.legend(loc=(1.0,0))
            fig.savefig(f'{pathFig}\\{fName}.png')
            plt.close(fig)
        except Exception as e:
            print(f'Error in plotting {fName}: \n\t{e}', flush=True)
        # try

        try:
            with ExcelWriter(f'{pathData}\\{fName}.xlsx') as ex_wrtr:
                self.data.to_excel(ex_wrtr, sheet_name=f'{nmSt}')
            # with
        except Exception as e:
            print(f'Error in creating {pm_name}.xlsx not created in {pathData} !', flush=True)
        # try    
    #

    def barPlot(self):
        """
        Draw a barplot
        """
        
        for pm_name in list(self.pmList): #pm_lst [auc, acc, ...]
            for nmSt in nameStat_gl:

                self.barPlotSub(self.pathplots_bar,pm_name,nmSt)

            #for nmSt
        #for pm_name
    #
        


