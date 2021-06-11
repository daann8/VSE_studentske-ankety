#Library import
from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
import re

from pandas.core.frame import DataFrame
#Load data
df = pd.read_excel(r"C:\Users\zbyne\Documents\čokoláda\VŠE\datový projekt\Textová analytika - Projekt\odpovedi_dotaznik.xlsx")
#understand data and do necessary changes
df.shape
df.head(13)
df["HODNOTA_CLOB"].dropna(inplace=True)
df['HODNOTA_CLOB'] = df['HODNOTA_CLOB'].astype('string')
df.dtypes

#Only rows that are not NA
df = df[df['HODNOTA_CLOB'].notna()]

#simple regexes x
#nums
def is_num(inp):
    pattern = '^\d+'
    x = re.findall(pattern,inp)
    return x

#extract "&quot; smt &quot"
def erase_quot(inp):
    pattern = '\&quot;'
    x = re.sub(pattern,'',inp)
    return x

def erase_x000D(inp):
    pattern = '\_x000D_'
    x = re.sub(pattern,'',inp)
    return x

#Apply functions on df
df['HODNOTA_CLOB_CISLA'] = df['HODNOTA_CLOB'].apply(is_num)
df["HODNOTA_CLOB_OCISTENO"] = df["HODNOTA_CLOB"].apply(erase_quot)
df["HODNOTA_CLOB_OCISTENO"] = df["HODNOTA_CLOB_OCISTENO"].apply(erase_x000D)

df["HODNOTA_CLOB_CISLA"] = df["HODNOTA_CLOB_CISLA"].apply(lambda x: np.nan if len(x)==0 else x[-1])

#change type of cols
df['HODNOTA_CLOB_CISLA'] = df['HODNOTA_CLOB_CISLA'].astype('float')
df['HODNOTA_CLOB_CISLA'] = df['HODNOTA_CLOB_CISLA'].astype('Int32')

#Get only answers "suitable for text analysis"
df_texts = df[df['HODNOTA_CLOB_CISLA'].isna()]
df_texts = df_texts[~df_texts["OTAZKA"].isin([21,42,92])]
df_texts.drop(columns=["HODNOTA_CLOB_CISLA"],inplace = True)

#According to Q assign sentiment of answer:
#Qs assuming negative answers = 4,57,83,95,45
#Qs assuming positive answers = 3,97,47,56,84
#Qs assuming int as answers = 21,42,92
#Qs assuming neutral answer = 54,23,90

#functions to add sentiment based on type of an question
def add_sentiment(i):
    negans = [4,57,83,95,45]
    posans = [3,97,47,56,84]
    #intans = [21,42,92]
    neuans = [54,23,90]

    if i in negans:
        return "Negative"
    elif i in posans:
        return "Positive"
    else:
        return "Neutral"

#add sentiment
df_texts["SENTIMENT"] = df_texts["OTAZKA"].apply(add_sentiment)

##########################################################################################################
df_to_export = df_texts[['ID','DOTAZNIK','OTAZKA','PEDAGOG','HODNOTA_CLOB_OCISTENO','SENTIMENT']]
df_to_export.to_csv(r'C:\Users\zbyne\Documents\čokoláda\VŠE\datový projekt\Textová analytika - Projekt\BaseCleanUp\Exported.csv')

predmety = pd.read_excel(r"C:\Users\zbyne\Documents\čokoláda\VŠE\datový projekt\Textová analytika - Projekt\predmety_kit.xlsx")
predmety_to_export = predmety[["ID","NAZEV"]]
predmety_to_export.to_csv(r'C:\Users\zbyne\Documents\čokoláda\VŠE\datový projekt\Textová analytika - Projekt\BaseCleanUp\predmety_kit.csv')

