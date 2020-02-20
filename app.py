from flask import Flask,flash,render_template,url_for,request,redirect

app = Flask(__name__)

app.secret_key = b'B\xdb\x1aF<\xb6\xcd\xce\x08\xe9Vw\xa4\xb9\xcd\xdb \xa6\x90G\xf4\xbf<\xf5'

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predictTrend', methods=["GET","POST"])
def predict():
        error = " "
        try:
            if request.method == "POST":
                name = request.form['keyword']
                flash(name)
                if name == "sahil":
                    return redirect('/predictTrend')
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
                flash(name)
                if name == "sahil":
                    return redirect('/predictUser')
                else:
                    error = "invalid credentials"
            
            return render_template('predictUser.html', error = error)
        
        except Exception as e:
            flash(e)
            return render_template('predictUser.html', error = error)           
    


if __name__ == '__main__':
	app.run(debug=True)
