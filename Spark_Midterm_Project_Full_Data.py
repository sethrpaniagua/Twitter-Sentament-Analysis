#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


PATH = r"C:\Users\cuong\2021 Machine Learning\data"
conf = SparkConf().setAppName("CIS5367 Midterm App").setMaster("local")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)
sql_context = SQLContext(sc)


# In[25]:


def get_data():
    twitter_data = sql_context.read.load("%s/twitter_dataset.csv" % PATH,    
                      format='com.databricks.spark.csv', 
                      header='true', 
                      inferSchema='true')
    #Delete all duplicated rows
    twitter_data = twitter_data.distinct()
    
    #Return a dataframe
    return twitter_data


# In[26]:


def visualize_tweet_data(twitter_data):
    from pyspark.sql.functions import avg, col, length
    from pyspark.sql.functions import lower, split
    import itertools
    import collections
    import seaborn as sns
    #import nltk
    from nltk.corpus import stopwords
    import pandas as pd
    from wordcloud import WordCloud
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    
    #Number of different tweets
    num_of_diftweet = len(twitter_data.groupBy("tweet").count().collect())
    print("Number of different tweets are: " + str(num_of_diftweet))
    
    #Average length of review text   
    text_length = twitter_data.withColumn("length", length(twitter_data["tweet"]))
    avg_length = round(text_length.select(avg("length")).collect()[0][0],2)
    print("Average length of text is: " + str(avg_length))
    
    #Make all twitter to lower case and split them
    words_in_tweet = twitter_data.select(split(lower(col("tweet"))," ")).collect()
    
    #This is a list contains many lists for all tweet
    """This list will use for applying stopword and collection word"""
    word_list = []
    for each_tweet in words_in_tweet:
        for word in each_tweet:
            word_list.append(word)

    #Eliminate stopwords to eliminate the common words
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    
    tweets_nsw = [[word for word in tweet_words if not word in stop_words]
        for tweet_words in word_list]
    
    #Eliminate collection words
    collection_words = ['gucci','polo','chanel','burberry','prada','versace','fendi','hermes','new','loving','never','check',                       'share','someone','fashion','got','played']
    
    tweets_nsw_nc = [[w for w in word if not w in collection_words]
        for word in tweets_nsw]
    #Create a list of words after cleaning all common words
    all_words_nsw_nc = list(itertools.chain(*tweets_nsw_nc))
    counts_nsw_nc = collections.Counter(all_words_nsw_nc)
    input_data = counts_nsw_nc
    
    #Plot word count function
    word_count_visualization(input_data)
    
    #Word cloud
    create_wordcloud(input_data)


# In[27]:


def create_wordcloud(input_data):
    from wordcloud import WordCloud
    #Word Cloud
    unique_string=(" ").join(input_data)
    wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig("your_file_name"+".png", bbox_inches='tight')
    plt.show()
    plt.close()


# In[28]:


def word_count_visualization(input_data):
    import pandas as pd
    
    input_data = pd.DataFrame(input_data.most_common(30),
                             columns=['words', 'count'])
    # Word Count visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    # Plot horizontal bar graph
    input_data.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="purple")
    ax.set_title("Common Words Found in Tweets (Without Stop or Collection Words)")
    plt.show()


# In[29]:


def visualize_polarity_allbrands(twitter_data):
    import matplotlib.pyplot as plt
    import pandas as pd
    polarity_data = twitter_data.select("polarity").collect()
    brand = 'all'
    #Call plot the histogram of polarity
    plot_polarity(polarity_data,brand)


# In[30]:


def visualize_brand(twitter_data):
    import pyspark.sql.functions as f
    
    brand_list = ('Gucci','Polo','Chanel','Burberry','Prada','Versace','Fendi','Hermes')
    
    for brand in brand_list:
        input_data = twitter_data.filter(f.col('brand_name')==brand)
        polarity_data = input_data.select("polarity").collect()
        #Call plot histogram function
        plot_polarity(polarity_data,brand)


# In[34]:


def plot_polarity(polarity_data,brand):
    import matplotlib.pyplot as plt
    import pandas as pd
    list_polarity_alldata = []
        
    for i in range(0,len(polarity_data)) :
        list_polarity_alldata.append(round(polarity_data[i][0],2))
        #print(list_polarity_alldata)

    polarity_df = pd.DataFrame(list_polarity_alldata, columns=["polarity"])
    polarity_df = polarity_df[polarity_df.polarity !=0]

    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot histogram of the polarity values
    polarity_df.hist(bins=[-0.5, -0.25, 0.25, 0.5, 0.75, 1],
            ax=ax)
            #color="purple")

    plt.title("Historgram of Sentiments from Tweets of " + brand + " brand")
    plt.show()


# In[35]:


def main():
    twitter_data = get_data()
    visualize_tweet_data(twitter_data)
    #visualize_polarity_allbrands(twitter_data)
    visualize_brand(twitter_data)


# In[36]:


main()
#sc.stop()


# In[ ]:




