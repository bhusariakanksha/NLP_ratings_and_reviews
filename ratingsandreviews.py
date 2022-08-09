# Databricks notebook source
storage_account_name = "pieratingsandreview"
storage_account_access_key = "YyCTTVQuuVVs+oVikAPTD9TYfoyNiCbb/aiT79cF27vh/HVfNFF4IEiGQiccZlWG2/4U2iOxAtRuvIS2BoqC9Q=="


spark.conf.set(
  "fs.azure.account.key."+storage_account_name+".blob.core.windows.net",
  storage_account_access_key)

# COMMAND ----------

import pandas as pd
from azure.storage.blob import BlockBlobService
blob_service=BlockBlobService(account_name=storage_account_name,account_key=storage_account_access_key)
import datetime
from dateutil import parser

# COMMAND ----------

file_type = "csv"
df=spark.read.format(file_type).option("inferSchema", True).option("header", True).load('wasbs://ratingsandreviews@pieratingsandreview.blob.core.windows.net/nlp.cue_result.csv')
df1 = spark.read.format(file_type).option("inferSchema", True).option("header", True).load('wasbs://ratingsandreviews@pieratingsandreview.blob.core.windows.net/nlp.cue_mapping.csv')

# COMMAND ----------

#NLP Cue results
display(df)

# COMMAND ----------

#Cue mapping
display(df1)

# COMMAND ----------

import pandas as pd

pdDF = df1.toPandas()

# COMMAND ----------

pdDF.head()

# COMMAND ----------

numOfKeys = pdDF['keywords'].str.len()
numOfKeys1 = numOfKeys.to_frame()
numOfKeys1.index.name = 'id'
numOfKeys1.columns = ["NumKeys"]
print(numOfKeys1)

# COMMAND ----------

topCat = pdDF['categorisation']
topCat1 = topCat.to_frame()
topCat1.index.name = 'id'
topCat1.columns = ["Categories"]
print(topCat1)

# COMMAND ----------

keyPerCat = numOfKeys1.join(topCat1)
print(keyPerCat)

# COMMAND ----------

keyPerCatSums = keyPerCat.groupby(['Categories']).sum()
print(keyPerCatSums)

# COMMAND ----------


ax = keyPerCat.plot.bar(x='Categories', y='NumKeys', rot=0, figsize=(44,20))
plt.xticks(rotation=45)

# COMMAND ----------


ax = keyPerCatSums.plot.bar(y='NumKeys', use_index=True, rot=0, figsize=(44,20))
plt.xticks(rotation=45)

# COMMAND ----------

# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "false")
result_pdf = df.select("*").toPandas()
#import pandas as pd
#final_df =df.toPandas()

# COMMAND ----------

import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)
display(result_pdf.head())

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

#cue_results_test = pd.read_csv("Downloads/cue_results_test.csv", error_bad_lines=False)
#print('data read in!')


graph1 = result_pdf['categorisation'].value_counts().plot(kind='bar',
                                    figsize=(24,8),
                                    title="Frequency of Categories")
graph1.set_xlabel("Categories")
graph1.set_ylabel("Count")

plt.show()

# COMMAND ----------

result_pdf.isnull().sum()

# COMMAND ----------

(len(result_pdf) - result_pdf['categorisation'].count())/(len(result_pdf)) * 100

# COMMAND ----------

from matplotlib.ticker import StrMethodFormatter
import logging
import pandas as pd
import numpy as np
from numpy import random
#import gensim
import nltk
#nltk.download()
#nltk.download('all')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
#from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from functools import reduce
import pyspark
from pyspark.sql import SparkSession

# COMMAND ----------

file_type = "csv"
TaK = spark.read.format(file_type).option("inferSchema", True).option("header", True).load('wasbs://ratingsandreviews@pieratingsandreview.blob.core.windows.net/Themes&KeywordsFINAL.csv')

# COMMAND ----------

display(TaK)

# COMMAND ----------

TK = TaK.toPandas()
TK['Keywords'] = TK['Keywords'].str.lower()
display(TK)

# COMMAND ----------

subset = result_pdf[0:1000000]
display(subset)

# COMMAND ----------

subset1 = subset['content']
subsetx = subset[['content', 'brand_name', 'title', 'comment_time', 'online_score']].copy()
subset1 = subset1.to_frame()
#subsetx = subsetx.to_frame()
#print(subset1)
display(subsetx)

# COMMAND ----------

from string import printable


subset1 = subset1.mask(subset1.eq('None')).dropna()
#subsetx = subsetx.mask(subsetx.eq('None')).dropna()

subset1 = subset1.applymap(lambda y: ''.join(filter(lambda x: 
            x in printable, y)))
#subsetx = subsetx.applymap(lambda y: ''.join(filter(lambda x: 
#            x in printable, y)))

subset1['index1'] = subset1.index
subsetx['index1'] = subsetx.index

subset1 = subset1[subset1['content'] != ''] 
subsetx = subsetx[subsetx['content'] != ''] 

#subset1['content1'] = subset1['content'].str.rstrip("\r")

#for row in subset1.itertuples():
#  row.content = row.content.rstrip("\\r\\n")
#subset1 = string.replace("\r","")
#subset1['content1'] = subset1['content'].str.rstrip("\\r\\n")
subset1['content'] = subset1['content'].str.replace('\\', '0')
subset1['content'] = subset1['content'].str.replace('0r0n', ' ')
subsetx['content'] = subsetx['content'].str.replace('\\', '0')
subsetx['content'] = subsetx['content'].str.replace('0r0n', ' ')


#print(len(subsetx))
#display(subset1)
#display(subsetx)


# COMMAND ----------

display(subset1)

# COMMAND ----------

display(subsetx)

# COMMAND ----------

import nltk
nltk.download('punkt')
sents = []
index1 = []

#subset1['split_sentences'] = subset1.apply(lambda row: nltk.tokenize.sent_tokenize(row['content']), axis=1)

for row in subset1.itertuples():
  #row[1] = row[1].str.lower()
  sents.append(nltk.tokenize.sent_tokenize(row[1]))
  index1.append(row[2])


tester = pd.DataFrame(np.array(sents))
tester = tester.rename(columns={0: "split_sentences"})
tester['index1'] = index1
subset2 = tester.explode('split_sentences')
#subset2['Theme'] = 'insertthemehere'
subset2['split_sentences'] = subset2['split_sentences'].astype(str).str.lower()
#print(subset2)




# COMMAND ----------

display(subset2)


# COMMAND ----------

TK['split_keys'] = TK['Keywords'].str.split(",")
display(TK)

# COMMAND ----------


def categorize(text):
  maxcount = 0
  FinalSubTheme = 'Uncategorized'
  for x in range(63):
    count = 0
    SubTheme = ''
    for y in range(len(TK['split_keys'][x])):
      if TK['split_keys'][x][y] in text: 
        count =  count + 1
        SubTheme = TK['Sub Themes'][x]
    if count > maxcount:
      maxcount = count
      FinalSubTheme = SubTheme
  return (FinalSubTheme)
    
subset2['SubTheme'] = subset2.split_sentences.apply(categorize)

    

# COMMAND ----------

display(subset2)

# COMMAND ----------

X = (subset2['split_sentences'])
y = (subset2['SubTheme'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)

from sklearn.linear_model import SGDClassifier

sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=0, max_iter=5, tol=None)),
               ])
sgd.fit(X_train, y_train)


y_pred = sgd.predict(X_test)

print('Linear Support Vector Machine accuracy %s' % accuracy_score(y_pred, y_test))

# COMMAND ----------


ThemeList = []
HPEList = []
SentimentList = []
ThemeDict = dict(zip(TK['Sub Themes'], TK['Theme']))
HPEDict = dict(zip(TK['Sub Themes'], TK['HPE Theme']))
SentimentDict = dict(zip(TK['Sub Themes'], TK['Sentiment']))
#print(ThemeDict)
for index, row in subset2.iterrows():
    #print(row['split_sentences'], row['SubTheme'])
    ST = row['SubTheme']
    #print(ST)
    x = ThemeDict.get(ST)
    y = HPEDict.get(ST)
    z = SentimentDict.get(ST)
    print(x)
    ThemeList.append(x)
    HPEList.append(y)
    SentimentList.append(z)
    
subset2['Theme'] = ThemeList
subset2['HPE Theme'] = HPEList
subset2['Sentiment'] = SentimentList

display(subset2)

# COMMAND ----------

collapse1 = subset2.groupby('index1')['split_sentences'].apply(list).reset_index(name='Reviews')
collapse2 = subset2.groupby('index1')['SubTheme'].apply(list).reset_index(name='Sub Themes')
collapse3 = subset2.groupby('index1')['Theme'].apply(list).reset_index(name='Themes')
collapse4 = subset2.groupby('index1')['HPE Theme'].apply(list).reset_index(name='HPE Themes')
collapse5 = subset2.groupby('index1')['Sentiment'].apply(list).reset_index(name='Sentiments')

subset3 = collapse1.join(collapse2.set_index('index1'), on='index1')
subset3 = subset3.join(collapse3.set_index('index1'), on='index1')
subset3 = subset3.join(collapse4.set_index('index1'), on='index1')
subset3 = subset3.join(collapse5.set_index('index1'), on='index1')
print(len(subset3))
#display(subset3)

subset4 = pd.merge(subset3, subsetx, how='left', on=['index1', 'index1'])
display(subset4)
#print(len(subset4))


# COMMAND ----------

#spark.conf.set(
#  "fs.azure.account.key."+storage_account_name+".blob.core.windows.net",
#  storage_account_access_key)

#output_container_path = "wasbs://ratingsandreviews@pieratingsandreview.blob.core.windows.net/" 
#output_blob_folder = "%s" % output_container_path

# write the dataframe as a single file to blob storage
##(subset4
 #.coalesce(1)
 #.write
 #.mode("overwrite")
 #.option("header", "true")
 #.format("com.databricks.spark.csv")
# .to_csv(output_blob_folder))

#subset4.coalesce(1).write.format(“com.databricks.spark.csv”).option(“header”, #“true”).save(“wasbs://ratingsandreviews@pieratingsandreview.blob.core.windows.net/subset4.csv”)

output = subset4.to_csv(encoding = "utf-8")
blob_service.create_blob_from_text('ratingsandreviews', 'subset5.csv', output)

# Get the name of the wrangled-data CSV file that was just saved to Azure blob storage (it starts with 'part-')
#files = dbutils.fs.ls(output_blob_folder)
#output_file = [x for x in files if x.name.startswith("part-")]

# Move the wrangled-data CSV file from a sub-folder (wrangled_data_folder) to the root of the blob container
# While simultaneously changing the file name
#dbutils.fs.mv(output_file[0].path, "%s/predict-transform-output.csv" % output_container_path)


# COMMAND ----------

