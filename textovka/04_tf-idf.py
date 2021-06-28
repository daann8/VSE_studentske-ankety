import collections
import nltk
import pandas as pd
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\zbyne\Documents\čokoláda\VŠE\datový projekt\Textová analytika - Projekt\BaseCleanUp\CleanedLemmaToTf-Idf.csv')
df_positive = df[df["SENTIMENT"]=="Positive"]
df_negative = df[df["SENTIMENT"]=="Negative"]
#corpus = list(df['lemma_cleaned'])

vectorizer = TfidfVectorizer(stop_words=['nan','mr','ms'],encoding='utf-8')
#positive cloud
vecs = vectorizer.fit_transform(df_positive['lemma_cleaned'].values.astype('U'))
feature_names = vectorizer.get_feature_names()
dense = vecs.todense()
lst1 = dense.tolist()
dftf_positive = pd.DataFrame(lst1, columns=feature_names)

Tdf_positive = dftf_positive.T.sum(axis=1)
Tdf_positive.sort_values(ascending=False, inplace = True)

#first positive cloud
CloudPositive = WordCloud(background_color="white", max_words=100, stopwords=["nan","11","2pr101"]).generate_from_frequencies(Tdf_positive.sort_values(ascending=False))
plt.figure(figsize=(20,15))
plt.imshow(CloudPositive, interpolation='bilinear')
plt.axis("off")
plt.title("Positive")
plt.show()

#negative cloud
vecs = vectorizer.fit_transform(df_negative['lemma_cleaned'].values.astype('U'))
feature_names = vectorizer.get_feature_names()
dense = vecs.todense()
lst1 = dense.tolist()
dftf_negative = pd.DataFrame(lst1, columns=feature_names)

Tdf_negative = dftf_negative.T.sum(axis=1)
Tdf_negative.sort_values(ascending=False,inplace = True)

#first negative cloud
CloudNegative = WordCloud(background_color="black", max_words=100, stopwords=["nan","11","2pr101"]).generate_from_frequencies(Tdf_negative.sort_values(ascending=False)[1:])
plt.figure(figsize=(20,15))
plt.imshow(CloudNegative, interpolation='bilinear')
plt.axis("off")
plt.title("Negative")
plt.show()

#rank both series and if the word in positive better rank than the word in neg it will be showed
#in positive and other way around :)
#click on word -> get the comments with it.

#get only those which are different between
dftf_negative = Tdf_negative.to_frame()
dftf_positive = Tdf_positive.to_frame()
#rank negative
dftf_negative.reset_index(inplace=True)
dftf_negative.rename(columns={"index":"word",0:"importance"},inplace=True)
dftf_negative['rank'] = dftf_negative['importance'].rank(ascending = False)
#rank positive
dftf_positive.reset_index(inplace=True)
dftf_positive.rename(columns={"index":"word",0:"importance"},inplace=True)
dftf_positive['rank'] = dftf_negative['importance'].rank(ascending = False)

#compare length of df
def compare_length(df1,df2):
    l1 = len(df1)
    l2 = len(df2)
    minimum = min(l1,l2)

    if minimum == l1:
        return l1
    else:
        return l2

uniquepositivewords = []
uniquenegativewords = []
#compare thos ranks between two df
for i in range(compare_length(dftf_negative,dftf_positive)):
    positiveword = dftf_positive.iloc[i,0]
    positivewordrank = dftf_positive.iloc[i,2]

    try:
        samenegativeword = dftf_negative.loc[dftf_negative['word'] == positiveword].iloc[0,0]
        negativewordrank = dftf_negative.loc[dftf_negative['word'] == positiveword].iloc[0,2]
    except IndexError:
        #word is not in this negative dataset --> can go straight into positivewords if there is corresponding value
        uniquepositivewords.append(positiveword)
    
    if positivewordrank < negativewordrank:
         uniquepositivewords.append(positiveword)
    else:
        uniquenegativewords.append(samenegativeword)
#some last adjustments
dfforwcloudPositive = dftf_positive[dftf_positive["word"].isin(set(uniquepositivewords))]
dfforwcloudPositive.reset_index(inplace=True,drop = True)
dfforwcloudPositive.set_index("word",inplace=True)

dfforwcloudNegative = dftf_negative[dftf_negative["word"].isin(set(uniquenegativewords))]
dfforwcloudNegative.reset_index(inplace=True,drop = True)
dfforwcloudNegative.set_index("word",inplace=True)

#rework to pandas core series, can't feed with pandas df.

#final negative cloud
CloudNegative = WordCloud(background_color="black", max_words=50, stopwords=["nan","11","2pr101"]).generate_from_frequencies(dfforwcloudNegative.iloc[:,0])
plt.figure(figsize=(20,15))
plt.imshow(CloudNegative, interpolation='bilinear')
plt.axis("off")
plt.title("Negative")
plt.show()
#final positive cloud
CloudPositive = WordCloud(background_color="white", max_words=50, stopwords=["nan","11","2pr101"]).generate_from_frequencies(dfforwcloudPositive.iloc[:,0])
plt.figure(figsize=(20,15))
plt.imshow(CloudPositive, interpolation='bilinear')
plt.axis("off")
plt.title("Positive")
plt.show()
