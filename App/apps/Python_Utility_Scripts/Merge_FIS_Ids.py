import pandas as pd
import numpy as np
import pathlib
import os


##############################################################
# Dash requires pathlib standardized paths for reading
# instructions the primary used directories are declared here
# ############################################################

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../../Data").resolve()
FIS_DATA_PATH = PATH.joinpath("../../Data/FIS Data").resolve()

def run():
    df = pd.read_csv(DATA_PATH.joinpath('Alpine_Skiing_Cleaned.csv'))
    cwd = os.getcwd()
    os.chdir(FIS_DATA_PATH)
    new_dir = os.getcwd()
    onlyfiles = [os.path.join(new_dir, f) for f in os.listdir(new_dir) if os.path.isfile(os.path.join(new_dir, f))]
    os.chdir(cwd)
    fis_df_list = []
    for i in range(0,len(onlyfiles)):
        fis_df_i = pd.read_csv(onlyfiles[i])
        fis_df_list.append(fis_df_i)
    fis_df = pd.concat(fis_df_list)
    fis_df = fis_df[fis_df['Gender'].isin(['M','W'])]
    gender_converter = {"M":"Men","W":"Women"}
    fis_df['Lastname'] = fis_df.apply(lambda x: str(x['Lastname']).lower(),axis=1)
    fis_df['Firstname'] = fis_df.apply(lambda x: str(x['Firstname']).lower(),axis=1)
    fis_df['Fullname'] = fis_df.apply(lambda x: str(x['Firstname']) + ' ' + str(x['Lastname']) ,axis=1)
    fis_df['Gender'] = fis_df.apply(lambda x: gender_converter[x['Gender']],axis=1)
    fis_df =fis_df[['Competitorid','Fullname','Nationcode','Gender']]
    df['Namelower'] = df.apply(lambda x: x['Competitor'].lower(),axis=1)
    df = df[['Namelower','Event Gender','NOC']]
    fis_merged = df.merge(fis_df,how='left',left_on=['Namelower','Event Gender','NOC'],right_on=['Fullname','Gender','Nationcode'])
    fis_merged.dropna(inplace=True)
    fis_merged['Competitorid'] = fis_merged['Competitorid'].astype(np.int32)
    fis_merged.reset_index(inplace=True)
    fis_merged = fis_merged[['Competitorid','Fullname','Nationcode','Gender']]
    fis_merged = fis_merged.drop_duplicates()
    fis_merged.to_csv(DATA_PATH.joinpath('FIS_CompID_XREF.csv'),index=False)


if __name__ == '__main__':
    run()