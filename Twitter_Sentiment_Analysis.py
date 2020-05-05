from tweepy import API 
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import joblib

 
import twitter_credentials

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics import classification_report

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


trainData = pd.read_csv("train.csv")
testData = pd.read_csv("test.csv")

vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)

train_vectors = vectorizer.fit_transform(trainData['Content'])
test_vectors = vectorizer.transform(testData['Content'])

# # # # Using SVM # # # #

#classifier_linear = svm.SVC(kernel='linear')
#classifier_linear.fit(train_vectors, trainData['Label'])
#prediction_linear = classifier_linear.predict(test_vectors)
#report = classification_report(testData['Label'], prediction_linear, output_dict=True)

classifier_linear = joblib.load("svm.pkl")

# # # # # # # # # # # # # 

# # # # Using KNN # # # #

#Y = trainData[['Label']]
#Y = Y['Label'].values
#k = 8  
#neigh = KNeighborsClassifier(n_neighbors = k).fit(train_vectors,Y)
neigh = joblib.load("knn.pkl")



# # # # # # # # # # # # #

# # # # Using logistic regression # # # #

#Y = np.asarray(trainData['Label'])
#Logistic = LogisticRegression(C=0.01, solver='liblinear').fit(train_vectors,Y)
Logistic = joblib.load("lr.pkl")


# # # # # # # # # # # # #
# # # # Using Tree Classifier # # # #

#Y = trainData[['Label']]
#sentTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
#sentTree.fit(train_vectors,Y)
sentTree = joblib.load("tc.pkl")


# # # # # # # # # # # # #
# # # # TWITTER CLIENTS # # # #
class TwitterClient():
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)

        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client
    
    def get_tweets(self, query, count = 10):
        tweets1 = []
        try: 
            # call twitter api to fetch tweets 
            fetched_tweets = self.twitter_client.search(q = query, count = count)
            
            for tweet in fetched_tweets: 
                
                # appending parsed tweet to tweets list 
                if not hasattr(tweet,'retweeted_status'): 
                    # if tweet has retweets, ensure that it is appended only once 
                    tweets1.append(tweet) 
                 
  
            # return parsed tweets 
            return tweets1
            
        except tweepy.TweepError as e: 
            # print error (if any) 
            print("Error : " + str(e)) 
    """
    def get_user_timeline_tweets(self, num_tweets):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(num_tweets):
            tweets.append(tweet)
        return tweets

    def get_friend_list(self, num_friends):
        friend_list = []
        for friend in Cursor(self.twitter_client.friends, id=self.twitter_user).items(num_friends):
            friend_list.append(friend)
        return friend_list

    def get_home_timeline_tweets(self, num_tweets):
        home_timeline_tweets = []
        for tweet in Cursor(self.twitter_client.home_timeline, id=self.twitter_user).items(num_tweets):
            home_timeline_tweets.append(tweet)
        return home_timeline_tweets

    """
# # # # TWITTER AUTHENTICATER # # # #
class TwitterAuthenticator():

    def authenticate_twitter_app(self):
        auth = OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)
        auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
        return auth

# # # #  TWITTER ANALYZER# # # #
class TweetAnalyzer():
    """
    Functionality for analyzing and categorizing content from tweets.
    """

    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def analyze_sentiment_svm(self, tweet):
        review_vector = vectorizer.transform([tweet])
        return classifier_linear.predict(review_vector)
    
    def analyze_sentiment_knn(self, tweet):
        review_vector = vectorizer.transform([tweet])
        return neigh.predict(review_vector)
    
    def analyze_sentiment_logisticReg(self, tweet):
        review_vector = vectorizer.transform([tweet])
        return Logistic.predict(review_vector)
      
    def analyze_sentiment_TreeClassifier(self, tweet):
        review_vector = vectorizer.transform([tweet])
        return sentTree.predict(review_vector)

    def tweets_to_data_frame(self, tweets):
        df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['tweets'])

        #df['id'] = np.array([tweet.id for tweet in tweets])
        df['len'] = np.array([len(tweet.text) for tweet in tweets])
        #df['date'] = np.array([tweet.created_at for tweet in tweets])
        #df['source'] = np.array([tweet.source for tweet in tweets])
        #df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
        #df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])

        return df

 
if __name__ == '__main__':

    twitter_client = TwitterClient()
    tweet_analyzer = TweetAnalyzer()

    api = twitter_client.get_twitter_client_api()
    
    #tweets = api.user_timeline(screen_name = "guardiannews", count=200)
    tweets1 = twitter_client.get_tweets(query = 'Dark Phoenix', count = 200)

    df = tweet_analyzer.tweets_to_data_frame(tweets1)
    df['sent_svm'] = np.array([tweet_analyzer.analyze_sentiment_svm(tweet) for tweet in df['tweets']])
    df['sent_knn'] = np.array([tweet_analyzer.analyze_sentiment_knn(tweet) for tweet in df['tweets']])
    df['sent_lr'] = np.array([tweet_analyzer.analyze_sentiment_logisticReg(tweet) for tweet in df['tweets']])
    df['sent_treeC'] = np.array([tweet_analyzer.analyze_sentiment_TreeClassifier(tweet) for tweet in df['tweets']])

    print(df.head(14))
    print('positive: ', report['pos'])
    print('negative: ', report['neg'])