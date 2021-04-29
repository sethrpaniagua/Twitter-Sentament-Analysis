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
sns.set(font_scale=1.5)
sns.set_style("whitegrid")


PATH = r"C:\Users\dk412\Desktop\TX State\CIS 5367 Machine Learning\Group Project"
conf = SparkConf().setAppName("CIS5367 Midterm App").setMaster("local")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)
sql_context = SQLContext(sc)

def get_data():
    twitter_data = sql_context.read.load("%s/twitter_10data.csv" % PATH,    
                      format='com.databricks.spark.csv', 
                      header='true', 
                      inferSchema='true')
    #Delete all duplicated rows
    #twitter_data = twitter_data.distinct()
    
    #Return a dataframe
    return twitter_data


########################### Getting the Dataset
twitter_data = get_data()

from pyspark.sql.functions import when 
twitter_data = twitter_data.withColumn('polarity_type', when(twitter_data.polarity > 0, 'positive')\
                                         .when(twitter_data.polarity < 0, 'negative').otherwise("neutral"))
    
    
import pandas as pd
import re
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from nltk.stem.snowball import SnowballStemmer
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql.functions import udf, lit
from pyspark.sql.types import StringType, ArrayType


(df_train_raw, df_test_raw) = twitter_data.randomSplit([0.8, 0.2], seed = 1234)
print (df_train_raw.first())

print ("\nTest Count::::: " , df_test_raw.count())
print ("\nTrain Count::::: " , df_train_raw.count())


df_train_raw = df_train_raw.withColumn("part", lit(1))
df_test_raw = df_test_raw.withColumn("part", lit(0))
df_all_raw = df_train_raw.union(df_test_raw)
df_all_raw.show()

############################## Preprocess Data

#Preprocessing Data
tokenizer = Tokenizer(inputCol='tweet', outputCol='words')
wordsData = tokenizer.transform(df_all_raw). \
    select('polarity_type', 'brand_name', 'words', 'part')
    
# Remove number
filter = re.compile(r"^[a-zA-Z]+$")
match_udf = udf(lambda tokens: [token for token in tokens if filter.match(token)], ArrayType(StringType()))
df_matched = wordsData.withColumn("words_matched", match_udf("words")). \
    select('words_matched', 'polarity_type', 'brand_name','part')
# Remove stop words
remover = StopWordsRemover(inputCol='words_matched', outputCol='words_clean')
df_words_clean = remover.transform(df_matched). \
    select('words_clean', 'polarity_type', 'part','brand_name')
#print(df_words_clean.show(5))
# Stem text
stemmer = SnowballStemmer(language='english')
stemmer_udf = udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType()))
df_stemmed = df_words_clean.withColumn("words_stemmed", stemmer_udf("words_clean")). \
    select('words_stemmed', 'polarity_type', 'part','brand_name')
    
#df_stemmed.show(2, truncate=False)

from pyspark.ml.feature import StringIndexer
# tf-idf
hashingTF = HashingTF(inputCol="words_stemmed", outputCol="rawFeatures") #generate vectors
featurizedData = hashingTF.transform(df_stemmed)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
#rescaledData.show()

#Add Brand_name -- group_index to the dataset
brandname_indexer = StringIndexer(inputCol="brand_name", outputCol="brandname_index")
df_brandname_indexed = brandname_indexer.fit(rescaledData).transform(rescaledData)

#Add polarity_type --polarity_index to the dataset
polarity_indexer = StringIndexer(inputCol="polarity_type", outputCol="polarity_index")
df_two_indexed = polarity_indexer.fit(df_brandname_indexed).transform(df_brandname_indexed) 
df_two_indexed.show()

#Input features for Baynesian Methods is the combination of word counts and sentiments
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=["rawFeatures","polarity_index"],outputCol="final_features")
assemblerData = assembler.transform(df_two_indexed)
assemblerData.show()


df_train = assemblerData.where("part = 1")
df_test = assemblerData.where("part = 0")
df_test.show()


############################# CLASSIFYING


########## First Model Brand Name - Final Features
# Naive Bayes classifier- first model
from pyspark.ml.classification import NaiveBayes
   
nb = NaiveBayes(labelCol="brandname_index",\
    featuresCol="final_features", smoothing=1.0,\
    modelType="multinomial")
model = nb.fit(df_train)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
predictions = model.transform(df_test)
predictions.select("polarity_type","brand_name", "brandname_index", 
    "probability", "prediction").show()
evaluator =\
    MulticlassClassificationEvaluator(labelCol="brandname_index",\
    predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))

########## Second Model Polarity Index - Final Features

#Input features for Baynesian Methods is the combination of word counts and sentiments
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=["rawFeatures","polarity_index"],outputCol="final_features")
assemblerData = assembler.transform(df_two_indexed)
#assemblerData.show()

df_train = assemblerData.where("part = 1")
df_test = assemblerData.where("part = 0")
df_test.show()


# Naive Bayes classifier- first model
from pyspark.ml.classification import NaiveBayes
   
nb = NaiveBayes(labelCol="polarity_index",\
    featuresCol="final_features", smoothing=1.0,\
    modelType="multinomial")
model = nb.fit(df_train)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
predictions = model.transform(df_test)
predictions.select("polarity_type","brand_name", "polarity_index", 
    "probability", "prediction").show()
evaluator =\
    MulticlassClassificationEvaluator(labelCol="polarity_index",\
    predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))

########## Third Model Polarity Index - Features
# Naive Bayes classifier- first model
from pyspark.ml.classification import NaiveBayes
   
nb = NaiveBayes(labelCol="polarity_index",\
    featuresCol="features", smoothing=1.0,\
    modelType="multinomial")
model = nb.fit(df_train)


from pyspark.ml.evaluation import MulticlassClassificationEvaluator
predictions = model.transform(df_test)
predictions.select("polarity_type","brand_name", "polarity_index", 
    "probability", "prediction").show()
evaluator =\
    MulticlassClassificationEvaluator(labelCol="polarity_index",\
    predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))

########## Fourth Model Brand Name Index - Features
# Naive Bayes classifier- first model
from pyspark.ml.classification import NaiveBayes
   
nb = NaiveBayes(labelCol="brandname_index",\
    featuresCol="features", smoothing=1.0,\
    modelType="multinomial")
model = nb.fit(df_train)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
predictions = model.transform(df_test)
predictions.select("polarity_type","brand_name", 
    "probability", "prediction").show()
evaluator =\
    MulticlassClassificationEvaluator(labelCol="brandname_index",\
    predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))
predictions.show()

################### Brand Prediciton Averages and Plots

import matplotlib.pyplot as plt

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
hermes_subdf.count()

bins, counts = hermes_subdf.select('prediction').rdd.flatMap(lambda x: x).histogram(6)
plt.title("Twitter Sentiment Analysis: Hermes")
plt.xlabel("Sentiment Prediction")
plt.ylabel("# of Tweets")
plt.hist(bins[:-1], bins=bins, weights=counts)

