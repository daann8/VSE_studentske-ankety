#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import re
#load data
data = pd.read_csv(r"C:\Users\zbyne\Documents\čokoláda\VŠE\datový projekt\Textová analytika - Projekt\BaseCleanUp\EnglishLemmaCloud.csv")
#explore
data.shape
data.head()

sns.set_theme(style="darkgrid")
sns.countplot(x = "LANG", data = data)
sns.countplot(x = "DetectSentiment", data = data)
sns.countplot(x = "SENTIMENT", data = data)
#fix format of detected sentiment
NewTransDetSent = []
for i in range(len(data)):
    x = (data.iloc[i,9][2:10])
    NewTransDetSent.append(x)

data["DetectSentiment"] = NewTransDetSent
#prepare data for comparing detected vs real sent
PosNegData = data[~data["SENTIMENT"].isin(["Neutral"])]
SentimentDifference = []

#add column with 1 or 0, 1 sentiment matches 0 not.
for i in range(len(PosNegData)):
    if PosNegData.iloc[i,9] == PosNegData.iloc[i,10].lower():
        x = 1
        SentimentDifference.append(x)
    else:
        x = 0
        SentimentDifference.append(x)

PosNegData["Match"] = SentimentDifference
#show the results
sns.countplot(x = "Match", data = PosNegData)

#regex for replacing brackets in columns
def ReplaceBrackets(imp):
    pattern = r'\[(?:[^\]])?([^\]]*)\]'
    x = re.sub(pattern,r'\1',imp)
    return x
#apply function

data["lemma_cleaned"] = data["lemma"].apply(ReplaceBrackets)
data["token_cleaned"] = data["token"].apply(ReplaceBrackets)
data["pos_cleaned"] = data["pos"].apply(ReplaceBrackets)
#replace parenthesis

def SingleParenth(imp):
    return imp.replace("'", '')

#apply the function on cols
data["lemma_cleaned"] = data["lemma_cleaned"].apply(SingleParenth)
data["token_cleaned"] = data["token_cleaned"].apply(SingleParenth)
data["pos_cleaned"] = data["pos_cleaned"].apply(SingleParenth)

#unknown to cs

#[], ['sl'], ['pl'],['sk'] drop--> empty
#['sl']           41 #drop gramatic masterpiece or SK
#['pl']           16 #pl is new sk
#['hu']              #to cs

#drop most of sk lang -> prepare for tf-idf
data = data[~data["LANG"].isin(["['sl']","['pl']","['sk']"])]
#save
data.to_csv(r'C:\Users\zbyne\Documents\čokoláda\VŠE\datový projekt\Textová analytika - Projekt\BaseCleanUp\CleanedLemmaToTf-Idf.csv')
