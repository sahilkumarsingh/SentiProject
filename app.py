from flask import Flask,flash,render_template,url_for,request,redirect
import numpy as np
import matplotlib.pyplot as plt
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
                df['sent_knn'] = np.array([tweet_analyzer.analyze_sentiment_knn(tweet) for tweet in df['tweets']])
                df['sent_svm'] = np.array([tweet_analyzer.analyze_sentiment_svm(tweet) for tweet in df['tweets']])
                df['sent_lr'] = np.array([tweet_analyzer.analyze_sentiment_logisticReg(tweet) for tweet in df['tweets']])
                df['sent_treeC'] = np.array([tweet_analyzer.analyze_sentiment_TreeClassifier(tweet) for tweet in df['tweets']])
                
                ####
                polarity = ['POS', 'NEG']
                explode = [0, 0]
                
                
                # # # Creating pie chart for visual comparisons # # #
                svm_pos = df['sent_svm'].value_counts('pos')
                svm_p = svm_pos['pos']*100
                result = [svm_p,100-svm_p]
                plt.pie(result, labels = polarity ,autopct='%1.2f%%', radius = 0.50, shadow = True,wedgeprops = {'edgecolor': 'black', 'linewidth': 0})
                plt.title("Support Vector Machine (SVM)")
                plt.axis('equal')
                plt.savefig('/Users/sahil/Desktop/sentiproject/static/images/svmpie1.png', transparent= True)
                plt.close()
                
                
                
                knn_pos = df['sent_knn'].value_counts('pos')
                knn_p = knn_pos['pos']*100
                result = [knn_p,100-knn_p]
                plt.pie(result, labels = polarity ,autopct='%1.2f%%', shadow = True, radius = 0.50, wedgeprops = {'edgecolor': 'black', 'linewidth': 0})
                plt.title("K Nearest Neighnor (KNN)")
                plt.axis('equal')
                plt.savefig('/Users/sahil/Desktop/sentiproject/static/images/knnpie1.png', transparent= True)
                plt.close()
                
                
                                
                lr_pos = df['sent_lr'].value_counts('pos')
                lr_p = lr_pos['pos']*100
                result = [lr_p,100-lr_p]
                plt.pie(result, labels = polarity ,autopct='%1.2f%%', shadow = True, radius = 0.50, wedgeprops = {'edgecolor': 'black', 'linewidth': 0})
                plt.title("Logistic Regression")
                plt.axis('equal')
                plt.savefig('/Users/sahil/Desktop/sentiproject/static/images/lrpie1.png', transparent= True)
                plt.close()
                
                
                
                treeC_pos = df['sent_treeC'].value_counts('pos')
                treeC_p = treeC_pos['pos']*100
                result = [treeC_p,100-treeC_p]
                plt.pie(result, labels = polarity ,autopct='%1.2f%%', shadow = True, radius = 0.50, wedgeprops = {'edgecolor': 'black', 'linewidth': 0})
                plt.title("Tree Classifier")
                plt.axis('equal')
                plt.savefig('/Users/sahil/Desktop/sentiproject/static/images/tcpie1.png', transparent= True)
                plt.close()
                
                
                
                
                
                ####
                
                
                if name :
                    return render_template('result.html', df=df.head(3), Name = name)
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
                df['sent_svm'] = np.array([tweet_analyzer.analyze_sentiment_svm(tweet) for tweet in df['tweets']])
                df['sent_knn'] = np.array([tweet_analyzer.analyze_sentiment_knn(tweet) for tweet in df['tweets']])
                df['sent_lr'] = np.array([tweet_analyzer.analyze_sentiment_logisticReg(tweet) for tweet in df['tweets']])
                df['sent_treeC'] = np.array([tweet_analyzer.analyze_sentiment_TreeClassifier(tweet) for tweet in df['tweets']])
                ####
                polarity = ['POS', 'NEG']
                explode = [0, 0]
                
                
                # # # Creating pie chart for visual comparisons # # #
                svm_pos = df['sent_svm'].value_counts('pos')
                svm_p = svm_pos['pos']*100
                result = [svm_p,100-svm_p]
                plt.pie(result, labels = polarity ,autopct='%1.2f%%', radius = 0.50, shadow = True,wedgeprops = {'edgecolor': 'black', 'linewidth': 0})
                plt.title("Support Vector Machine (SVM)")
                plt.axis('equal')
                plt.savefig('/Users/sahil/Desktop/sentiproject/static/images/svmpie1.png', transparent= True)
                plt.close()
                
                
                
                knn_pos = df['sent_knn'].value_counts('pos')
                knn_p = knn_pos['pos']*100
                result = [knn_p,100-knn_p]
                plt.pie(result, labels = polarity ,autopct='%1.2f%%', shadow = True, radius = 0.50, wedgeprops = {'edgecolor': 'black', 'linewidth': 0})
                plt.title("K Nearest Neighnor (KNN)")
                plt.axis('equal')
                plt.savefig('/Users/sahil/Desktop/sentiproject/static/images/knnpie1.png', transparent= True)
                plt.close()
                
                
                                
                lr_pos = df['sent_lr'].value_counts('pos')
                lr_p = lr_pos['pos']*100
                result = [lr_p,100-lr_p]
                plt.pie(result, labels = polarity ,autopct='%1.2f%%', shadow = True, radius = 0.50, wedgeprops = {'edgecolor': 'black', 'linewidth': 0})
                plt.title("Logistic Regression")
                plt.axis('equal')
                plt.savefig('/Users/sahil/Desktop/sentiproject/static/images/lrpie1.png', transparent= True)
                plt.close()
                
                
                
                treeC_pos = df['sent_treeC'].value_counts('pos')
                treeC_p = treeC_pos['pos']*100
                result = [treeC_p,100-treeC_p]
                plt.pie(result, labels = polarity ,autopct='%1.2f%%', shadow = True, radius = 0.50, wedgeprops = {'edgecolor': 'black', 'linewidth': 0})
                plt.title("Tree Classifier")
                plt.axis('equal')
                plt.savefig('/Users/sahil/Desktop/sentiproject/static/images/tcpie1.png', transparent= True)
                plt.close()
                
                
                
                
                
                ####
                if name :
                    return render_template('result.html', df=df.head(3), Name = name)
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
