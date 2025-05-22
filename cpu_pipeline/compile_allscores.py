module_name = 'compile_Allscores'

'''
Version: v1.3.1

Description:
   Compiles both fold scores and best scores and their visualizations

Authors:
    Dr. Omer Muhammet Soysal
    Iphy Kelvin

Date Created     : 03/14/2025
Date Last Updated: 03/14/2025

Doc:

Notes:

ToDO:
    Rename 'single'-> 'IPM'; 'overall'-> 'OPM'

'''


# CUSTOM IMPORTS
from compile_folds      import folds_Assembler
from compile_bestscores import compile_bestscore
from config_compile     import ConfigMainCmp, ConfigPMCmp
from dataAssembler      import plotAssembler
from ui                 import All_ui

# OTHER IMPORTS
from csv    import writer as csvW
from os     import makedirs as osMakedirs
from pickle import dump

# USER INTERFACE
pathOutput_ui = All_ui['pathOutput_ui']

# CONSTANTS
pmName_IPM_gl = ConfigMainCmp['pmName_IPM_gl']


def main():

    # Run compile folds
    score_fold, errorLog = folds_Assembler()

    # Export
    pathScore = f'{pathOutput_ui}\\Foldscores_Assembled'; osMakedirs(pathScore, exist_ok=True)       
    score_fold.to_csv(f'{pathScore}\\score_fold.csv')
    
    with (open(f'{pathScore}\\scores_fold.pckl','wb') as pf1,
          open(f'{pathScore}\\errLog_fold_score.pckl','wb') as pf2):
        dump(score_fold, pf1)
        dump(errorLog, pf2)
    # with

    with open(f'{pathScore}\\errLog_fold.csv', 'w', newline='') as cf: 
        csvWr = csvW(cf)
        csvWr.writerow(['Key', 'Value'])

        for key, value in errorLog.items():
            csvWr.writerow([key, value])
        # for
    # with

    plotAssembler([score_fold], [pmName_IPM_gl], ['PM'], pathScore)  # pmCatgry: IPM

    # Run compile best
    scores_single, scores_overall, errorLog = compile_bestscore(foldername='StatsRepr')

    # Export
    pathScore = f'{pathOutput_ui}\\Bestscores_Assembled'; osMakedirs(pathScore, exist_ok=True)     
    
    with (open(f'{pathScore}\\IPM\\scores_single.pckl','wb') as pf1,
          open(f'{pathScore}\\OPM\\scores_overall.pckl', 'wb') as pf2,
          open(f'{pathScore}\\errLog_score.pckl','wb') as pf3):
        dump(scores_single, pf1)
        dump(scores_overall, pf2)
        dump(errorLog, pf3)
    #
    scores_single.to_csv(f'{pathScore}\\IPM\\scores_single.csv')
    scores_overall.to_csv(f'{pathScore}\\OPM\\scores_overall.csv')

    with open(f'{pathScore}\\errLog_score.csv', 'w', newline='') as cf: 
        csvWr = csvW(cf)
        csvWr.writerow(['Key', 'Value'])

        for key, value in errorLog.items():
            csvWr.writerow([key, value])
        # for
    # with

    plotAssembler(score_fold, pmName_IPM_gl, ['IPM'])  # pmCatgry: IPM



if __name__ == "__main__":
    pass
#

print(f"\"{module_name}\"module begins.")
main()
print('DONE', flush=True) 