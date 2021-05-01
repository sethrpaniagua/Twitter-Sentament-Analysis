# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 21:59:43 2021

@author: dk412
"""

from pyspark.sql.types import *
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, min, max
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.corpus import stopwords
import seaborn as sns
import pandas as pd
sns.set(font_scale=1.5)
sns.set_style("whitegrid")


PATH = r"C:\Users\dk412\Desktop\TX State\CIS 5367 Machine Learning\Group Project"
conf = SparkConf().setAppName("CIS5367 Midterm App").setMaster("local")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)
sql_context = SQLContext(sc)

def get_data():
    twitter_data = sql_context.read.load("%s/testdata.manual.2009.06.14.csv" % PATH,    
                      format='com.databricks.spark.csv', 
                      header='true', 
                      inferSchema= 'true',
                      encoding = 'utf-8')
    #Delete all duplicated rows
    #twitter_data = twitter_data.distinct()
    
    #Return a dataframe
    return twitter_data


########################### Getting the Dataset
#twitter_data = get_data().limit(10000)
twitter_data = get_data()
twitter_data.show()
twitter_data.count()
    
import pandas as pd
import re
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from nltk.stem.snowball import SnowballStemmer
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql.functions import udf, lit
from pyspark.sql.types import StringType, ArrayType


df_train_raw, df_test_raw = twitter_data.randomSplit([0.8, 0.2])
df_train_raw.show()
df_train_raw.count()

print ("\nTest Count::::: " , df_test_raw.count())
print ("\nTrain Count::::: " , df_train_raw.count())


df_train_raw = df_train_raw.withColumn("part", lit(1))
df_test_raw = df_test_raw.withColumn("part", lit(0))
df_all_raw = df_train_raw.union(df_test_raw)
df_all_raw.show()

############################## Preprocess Data

#Preprocessing Data
tokenizer = Tokenizer(inputCol='Text', outputCol='words')
wordsData = tokenizer.transform(df_all_raw). \
    select('polarity', 'words', 'part')
    
# Remove number
filter = re.compile(r"^[a-zA-Z]+$")
match_udf = udf(lambda tokens: [token for token in tokens if filter.match(token)], ArrayType(StringType()))
df_matched = wordsData.withColumn("words_matched", match_udf("words")). \
    select('words_matched', 'polarity','words','part')
    
# Remove stop words
remover = StopWordsRemover(inputCol='words_matched', outputCol='words_clean')
df_words_clean = remover.transform(df_matched). \
    select('words_clean', 'polarity', 'part','words')

# Stem text
stemmer = SnowballStemmer(language='english')
stemmer_udf = udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType()))
df_stemmed = df_words_clean.withColumn("words_stemmed", stemmer_udf("words_clean")). \
    select('words_stemmed', 'polarity', 'part')
    
df_stemmed.show(2)

# tf-idf
hashingTF = HashingTF(inputCol="words_stemmed", outputCol="rawFeatures") #generate vectors
featurizedData = hashingTF.transform(df_stemmed)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
rescaledData.show()

df_train = rescaledData.where("part = 1")
df_test = rescaledData.where("part = 0")
df_test.show()


############################# BUilding the Model CLASSIFYING

# Naive Bayes classifier- first model
from pyspark.ml.classification import NaiveBayes
   
nb = NaiveBayes(labelCol="polarity",\
    featuresCol="features", smoothing=1.0,\
    modelType="multinomial")
model = nb.fit(df_train)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
predictions = model.transform(df_test)
predictions.select("polarity","probability", "prediction").show()
evaluator =\
    MulticlassClassificationEvaluator(labelCol="polarity",\
    predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))

######################## Fashion Data
def get_fashion_data():
    twitter_fashion_data = sql_context.read.load("%s/twitter_10data.csv" % PATH,    
                      format='com.databricks.spark.csv', 
                      header='true', 
                      inferSchema= 'true',
                      encoding = 'utf-8')
    #Delete all duplicated rows
    #twitter_data = twitter_data.distinct()
    
    #Return a dataframe
    return twitter_fashion_data


########################### Getting the Dataset
#twitter_data = get_data().limit(10000)
twitter_fashion_data = get_fashion_data()
twitter_fashion_data.show()
twitter_fashion_data.count()


############################## Preprocess Data

#Preprocessing Data
tokenizer = Tokenizer(inputCol='tweet', outputCol='words')
wordsData = tokenizer.transform(twitter_fashion_data). \
    select('TweetID', 'brand_name','words')
    
# Remove number
filter = re.compile(r"^[a-zA-Z]+$")
match_udf = udf(lambda tokens: [token for token in tokens if filter.match(token)], ArrayType(StringType()))
df_matched = wordsData.withColumn("words_matched", match_udf("words")). \
    select('words_matched','words', 'TweetID', 'brand_name')
    
# Remove stop words
remover = StopWordsRemover(inputCol='words_matched', outputCol='words_clean')
df_words_clean = remover.transform(df_matched). \
    select('words_clean','words','TweetID', 'brand_name')

# Stem text
stemmer = SnowballStemmer(language='english')
stemmer_udf = udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType()))
df_stemmed = df_words_clean.withColumn("words_stemmed", stemmer_udf("words_clean")). \
    select('words_stemmed', 'TweetID', 'brand_name')
    
df_stemmed.show(2)

# tf-idf
hashingTF = HashingTF(inputCol="words_stemmed", outputCol="rawFeatures") #generate vectors
featurizedData = hashingTF.transform(df_stemmed)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
fashion_df_rescaled = idfModel.transform(featurizedData)
fashion_df_rescaled.show()

########## Perform Sentiment Analysis

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
predictions = model.transform(fashion_df_rescaled)
predictions.select("TweetID","brand_name","probability", "prediction").show()

################### Subset the data

####### POLO
polo_subdf = predictions.where("brand_name =='Polo'")
polo_subdf.select(avg("prediction")).show()
polo_subdf.show()
polo_subdf.count()

bins, counts = polo_subdf.select('prediction').rdd.flatMap(lambda x: x).histogram(6)
plt.title("Twitter Sentiment Analysis: Polo")
plt.xlabel("Sentiment Prediction")
plt.ylabel("# of Tweets")
plt.hist(bins[:-1], bins=bins, weights=counts)

####### GUCCI
gucci_subdf = predictions.where("brand_name =='Gucci'")
gucci_subdf.select(avg("prediction")).show()
gucci_subdf.show()
gucci_subdf.count()

bins, counts = gucci_subdf.select('prediction').rdd.flatMap(lambda x: x).histogram(6)
plt.title("Twitter Sentiment Analysis: Gucci")
plt.xlabel("Sentiment Prediction")
plt.ylabel("# of Tweets")
plt.hist(bins[:-1], bins=bins, weights=counts)

####### CHANEL
chanel_subdf = predictions.where("brand_name =='Chanel'")
chanel_subdf.select(avg("prediction")).show()
chanel_subdf.count()

bins, counts = chanel_subdf.select('prediction').rdd.flatMap(lambda x: x).histogram(6)
plt.title("Twitter Sentiment Analysis: Chanel")
plt.xlabel("Sentiment Prediction")
plt.ylabel("# of Tweets")
plt.hist(bins[:-1], bins=bins, weights=counts)

####### BURBERRY

burberry_subdf = predictions.where("brand_name =='Burberry'")
burberry_subdf.select(avg("prediction")).show()
burberry_subdf.show()
burberry_subdf.count()

bins, counts = burberry_subdf.select('prediction').rdd.flatMap(lambda x: x).histogram(6)
plt.title("Twitter Sentiment Analysis: Burberry")
plt.xlabel("Sentiment Prediction")
plt.ylabel("# of Tweets")
plt.hist(bins[:-1], bins=bins, weights=counts)

####### Prada

prada_subdf = predictions.where("brand_name =='Prada'")
prada_subdf.select(avg("prediction")).show()
prada_subdf.count()

bins, counts = prada_subdf.select('prediction').rdd.flatMap(lambda x: x).histogram(6)
plt.title("Twitter Sentiment Analysis: Prada")
plt.xlabel("Sentiment Prediction")
plt.ylabel("# of Tweets")
plt.hist(bins[:-1], bins=bins, weights=counts)

####### Versace

versace_subdf = predictions.where("brand_name =='Versace'")
versace_subdf.select(avg("prediction")).show()
versace_subdf.count()

bins, counts = versace_subdf.select('prediction').rdd.flatMap(lambda x: x).histogram(6)
plt.title("Twitter Sentiment Analysis: Versace")
plt.xlabel("Sentiment Prediction")
plt.ylabel("# of Tweets")
plt.hist(bins[:-1], bins=bins, weights=counts)

####### Fendi

fendi_subdf = predictions.where("brand_name =='Fendi'")
fendi_subdf.select(avg("prediction")).show()
fendi_subdf.count()

bins, counts = fendi_subdf.select('prediction').rdd.flatMap(lambda x: x).histogram(6)
plt.title("Twitter Sentiment Analysis: Fendi")
plt.xlabel("Sentiment Prediction")
plt.ylabel("# of Tweets")
plt.hist(bins[:-1], bins=bins, weights=counts)

####### Hermes

hermes_subdf = predictions.where("brand_name =='Hermes'")
hermes_subdf.select(avg("prediction")).show()
hermes_subdf.show()
hermes_subdf.count()

bins, counts = hermes_subdf.select('prediction').rdd.flatMap(lambda x: x).histogram(6)
plt.title("Twitter Sentiment Analysis: Hermes")
plt.xlabel("Sentiment Prediction")
plt.ylabel("# of Tweets")
plt.hist(bins[:-1], bins=bins, weights=counts)




