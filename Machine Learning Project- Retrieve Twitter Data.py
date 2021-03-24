#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# If you have not install tweepy, go to anaconda prompt
# conda install -c conda-forge tweepy
import tweepy as tw
import re

# If you have not install tweepy, go to anaconda prompt as administration mode and type
# conda install -c conda-forge textblob
from textblob import TextBlob

import warnings
warnings.filterwarnings("ignore")

access_token = "1142511379745779714-kb8dWmMfQveiGdZDGcWEQha3D3GRhy"

access_token_secret = "Vpo2SLbe3ipzMyat8SiPmML0ANS55IxwmJ8PoTuiSxW3B"

consumer_key = "YGdcKmAnYLm7z5xQ2lr6xxjQe"

consumer_secret = "NtM95IpGeYTf1WXKArLcBfIbPq7hP9lMjLD10a2D9840MblagR"

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)


# In[2]:


def remove_url(txt):
    """Replace URLs found in a text string with nothing
    (i.e. it will remove the URL from the string).

    Parameters
    ----------
    txt : string
        A text string that you want to parse and remove urls.

    Returns
    -------
    The same txt string with url's removed.
    """

    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())


# In[3]:


def get_tweet(keyword):
    search_term = "#" + keyword + " " + "-filter:tweets"

    tweets = tw.Cursor(api.search,
                   q=search_term,
                   lang="en",
                   since='2021-01-30').items(300)
    # Remove URLs
    tweets_no_urls = [remove_url(tweet.text) for tweet in tweets]

    # Create textblob objects of the tweets
    sentiment_objects = [TextBlob(tweet) for tweet in tweets_no_urls]
    #print(sentiment_objects[1])
    sentiment_objects[1].polarity, sentiment_objects[1]
    
    # Create list of polarity valuesx and tweet text
    sentiment_values = [[tweet.sentiment.polarity, str(tweet)] for tweet in sentiment_objects]
    sentiment_df = pd.DataFrame(sentiment_values, columns=["polarity", "tweet"])
    sentiment_df['brand_name'] = keyword
    return sentiment_df


# In[5]:


def main():
    keyword_list = ('Gucci','Polo','Chanel','Burberry','Prada','Versace','Fendi','Hermes')
    data_frame_list = []
    for keyword in keyword_list:
        data_frame = get_tweet(keyword)
        data_frame_list.append(data_frame)
    final_df = pd.concat(data_frame_list)
    final_df.reset_index(drop=True,inplace=True)
    final_df.to_csv(r'C:/Users/cuong/2021 Machine Learning/data/twitter_10data.csv',index=False)
    print("finish writting dataframe to the excel file")


# In[6]:


main()


# In[ ]:




