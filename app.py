from flask import Flask,flash,render_template,url_for,request,redirect
import numpy as np
import Twitter_Sentiment_Analysis

app = Flask(__name__)

app.secret_key = b'B\xdb\x1aF<\xb6\xcd\xce\x08\xe9Vw\xa4\xb9\xcd\xdb \xa6\x90G\xf4\xbf<\xf5'

@app.route('/')
def home():
	return render_template('index.html')
  
  
#@app.route('/results')
#def res():
#	return render_template('result.html')

@app.route('/predictTrend', methods=["GET","POST"])
def predict():
        error = " "
        try:
            if request.method == "POST":
                name = request.form['keyword']
                #flash(name)
                tweets1 = twitter_client.get_tweets(query = name, count = 200)
                df = tweet_analyzer.tweets_to_data_frame(tweets1)
                df['sentiment'] = np.array([tweet_analyzer.analyze_sentiment(tweet) for tweet in df['tweets']])
                if name :
                    return render_template('result.html', df=df.head(14))
                else:
                    error = "invalid credentials"
            
            return render_template('predictTrend.html', error = error)
        
        except Exception as e:
            flash(e)
            return render_template('predictTrend.html', error = error)
  

@app.route('/predictUser', methods=["GET","POST"])
def predict1():
        error = " "
        try:
            if request.method == "POST":
                name = request.form['username']
                api = twitter_client.get_twitter_client_api()
                #flash(name)
                tweets = api.user_timeline(screen_name = name, count=200)
                df = tweet_analyzer.tweets_to_data_frame(tweets)
                df['sentiment'] = np.array([tweet_analyzer.analyze_sentiment(tweet) for tweet in df['tweets']])
                if name :
                    return render_template('result.html', df=df.head(14))
                else:
                    error = "invalid credentials"
            
            return render_template('predictUser.html', error = error)
        
        except Exception as e:
            flash(e)
            return render_template('predictUser.html', error = error)           
    


if __name__ == '__main__':
    twitter_client = Twitter_Sentiment_Analysis.TwitterClient()
    tweet_analyzer = Twitter_Sentiment_Analysis.TweetAnalyzer()
    app.run(debug=True)
