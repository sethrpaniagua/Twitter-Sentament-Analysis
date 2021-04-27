# Twitter-Sentament-Analysis

## Project Descripition:
Develop and implement a machine learning sentiment analysis that analyzes large fashion brands Twitter data. The sentiment analysis will be used to make informed busienss decisions and help guide executives on fashion trends and marketing efforts. By utilizing a model that is trained to analyze sentiment around the highest trending fashion of major fashion brands and testing the model machine learning can better enhance the targeted marketing. 

## Use Case Description:
A local startup company of >75 employees that sells fast fashion items is looking to improve their targeted marketing to better tailor their products to the current fashion trends for the next quarter. To make an informed decision executive leadership has decide to contract a data analytics team to measure popular fashion trends using social media. The data analytics team has decided to develop a machine learning model that will test and train the twitter data of eight large fashion companies. The model will review the Twitter data of those large companies and perform a sentiment analysis that can be used to guide targeted marketing efforts. 

## Large Fashion Companies Twitter Handles for Analysis:
1. Gucci
2. Polo
3. Chanel
4. Burberry
5. Prada
6. Versace
7. Fendi
8. Hermes

## Useful Features:
Date Range: **2020-11-01 to Present**
Sampling: Index of **'10,000'** values
Dataset Columns:
- Polarity
- Tweet
- Fashion Brand Name

## Data Sources:
Conversion of live tweets through Twitter API to dataframe
Final data source is xlsx file - **twitter_10k_data.csv**

## Model Description:
The model uses a Naive Bayes classifier to learn and predict the sentiment of the data. The Naive Bayes classifier model is trained using the indexed polarity value as teh labelCol and the vectorized rescaled data as the featuresCol. The raw data is processed by undergoing tokenization, removal of numbers and stopwords, and stemming the data. The processed data is then hashed and rescaled using TF-IDF to vectorize and estimate. Then predicted values are tested for accuracy. 

## Model Evaluation:
Accuracy of the Model = 0.9145

## Conclusion:
Overall output and accuracy has met a satisfactory mark. Further work would include to develop a more robust training dataset or benchmark training set to make the model more accurate. The model that was built using Naive Bayes did have a higher accuracy then the intial use of TextBlob. 

