# -*- coding: utf-8 -*-

"""
Created on Sat Mar 13 11:17:04 2021

@author: sethr
"""
# Machine Learning Fast Fashion Project

# Import Libraries for use
import pandas as pd
import seaborn as sns
import tweepy as tw
import re
from textblob import TextBlob
import warnings


# Ignore Warnings
warnings.filterwarnings("ignore")
sns.set(font_scale=1.5)
sns.set_style("whitegrid")

# Define Access
access_token = "1142511379745779714-kb8dWmMfQveiGdZDGcWEQha3D3GRhy"

access_token_secret = "Vpo2SLbe3ipzMyat8SiPmML0ANS55IxwmJ8PoTuiSxW3B"

consumer_key = "YGdcKmAnYLm7z5xQ2lr6xxjQe"

consumer_secret = "NtM95IpGeYTf1WXKArLcBfIbPq7hP9lMjLD10a2D9840MblagR"

# Set Authorizations
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

# Function for taking away the url
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


# Function for getting tweets
def get_tweets(term):
    search_term = term
    tweets = tw.Cursor(api.search,
                       q=search_term,
                       lang="en",
                       since='2020-11-01').items(300)
    return tweets

gucci = get_tweets("#gucci -filter:tweets")
polo = get_tweets("#polo -filter:tweets")
chanel = get_tweets("#chanel -filter:tweets")
burberry = get_tweets("#burberry -filter:tweets")
prada = get_tweets("#prada -filter:tweets")
versace = get_tweets("#versace -filter:tweets")
fendi = get_tweets("#fendi -filter:tweets")
hermes = get_tweets("#hermes -filter:tweets")
                    


# Function for Removing Twitter URLS
def tweet_no_url(tweets):
    tweets_no_urls = [remove_url(tweet.text) for tweet in tweets]
    return tweets_no_urls

no_url_polo = tweet_no_url(polo)
no_url_gucci = tweet_no_url(gucci)
no_url_chanel = tweet_no_url(chanel)
no_url_burberry = tweet_no_url(burberry)
no_url_prada = tweet_no_url(prada)
no_url_versace = tweet_no_url(versace)
no_url_fendi = tweet_no_url(fendi)
no_url_hermes = tweet_no_url(hermes)

# Create textblob objects of the tweets
#sentiment_objects = [TextBlob(tweet) for tweet in tweets_no_urls]
def get_sentiment_objects(tweets_no_url):
    sentiment_objects = [TextBlob(tweet) for tweet in tweets_no_url]
    return sentiment_objects

polo_sentiment = get_sentiment_objects(no_url_polo)
gucci_sentiment = get_sentiment_objects(no_url_gucci)
chanel_sentiment = get_sentiment_objects(no_url_chanel)
burberry_sentiment = get_sentiment_objects(no_url_burberry)
prada_sentiment = get_sentiment_objects(no_url_prada)
versace_sentiment = get_sentiment_objects(no_url_versace)
fendi_sentiment = get_sentiment_objects(no_url_fendi)
hermes_sentiment = get_sentiment_objects(no_url_hermes)

    

# Create list of polarity values and tweet text
sentament_gucci = [[tweet.sentiment.polarity, str(tweet)] for tweet in gucci_sentiment]
sentament_polo = [[tweet.sentiment.polarity, str(tweet)] for tweet in polo_sentiment]
sentament_chanel = [[tweet.sentiment.polarity, str(tweet)] for tweet in chanel_sentiment]
sentament_burberry = [[tweet.sentiment.polarity, str(tweet)] for tweet in burberry_sentiment]
sentament_prada = [[tweet.sentiment.polarity, str(tweet)] for tweet in prada_sentiment]
sentament_versace = [[tweet.sentiment.polarity, str(tweet)] for tweet in versace_sentiment]
sentament_fendi = [[tweet.sentiment.polarity, str(tweet)] for tweet in fendi_sentiment]
sentament_hermes = [[tweet.sentiment.polarity, str(tweet)] for tweet in hermes_sentiment]


# Creation of dataframes depending on keywords of interest
sentament_guccidf = pd.DataFrame(sentament_gucci, columns=["polarity_gucci", "tweet_gucci"])
chaneldf = pd.DataFrame(sentament_chanel, columns=["polarity_chanel", "tweet_chanel"])
burberrydf = pd.DataFrame(sentament_burberry, columns=["polarity_burberry", "tweet_burberry"])
pradadf = pd.DataFrame(sentament_prada, columns=["polarity_prada", "tweet_prada"])
versacedf = pd.DataFrame(sentament_versace, columns=["polarity_versace", "tweet_versace"])
fendidf = pd.DataFrame(sentament_fendi, columns=["polarity_fendi", "tweet_fendi"])
hermesdf = pd.DataFrame(sentament_hermes, columns=["polarity_hermes", "tweet_hermes"])

# Tweet Objects Created
gucci_tweets = sentament_guccidf['tweet_gucci']
chanel = chaneldf["tweet_chanel"]
burberry = burberrydf["tweet_burberry"]
prada = pradadf["tweet_prada"]
versace = versacedf["tweet_versace"]
fendi = fendidf["tweet_fendi"]
hermes = hermesdf["tweet_hermes"]

# Polarity objects created
chanelp = chaneldf["polarity_chanel"]
burberryp = burberrydf["polarity_burberry"]
pradap = pradadf["polarity_prada"]
versacep = versacedf["polarity_versace"]
fendip = fendidf["polarity_fendi"]
hermesp = hermesdf["polarity_hermes"]


# Bring together all keywords
final_df= pd.DataFrame(sentament_polo, columns=["polarity_polo", "tweet_polo"])
final_df = final_df.join(gucci_tweets)
final_df = final_df.join(chanelp)
final_df = final_df.join(chanel)
final_df = final_df.join(burberryp)
final_df = final_df.join(burberry)
final_df = final_df.join(pradap)
final_df = final_df.join(prada)
final_df = final_df.join(versacep)
final_df = final_df.join(versace)
final_df = final_df.join(fendip)
final_df = final_df.join(fendi)
final_df = final_df.join(hermesp)
final_df = final_df.join(hermes)

# create excel writer object
writer = pd.ExcelWriter('fastfashion_datasource.xlsx')
# write and save to excel file for creation of datasource
final_df.to_excel(writer)
writer.save()


print("Program finished successfully!!!!")



