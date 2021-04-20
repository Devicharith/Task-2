from flask import Flask, request, render_template
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from topic_modelling import Topic_modeling


# create an instance
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Set the URL for the Function
@app.route('/output' , methods = ["POST"])
def TOP10():
    if request.method == 'POST':
        title = request.form['name']
        text = request.form['para']
        out = sent_tokenize(text)
        #segmentation
        for i in range(len(out)): out[i] = out[i].replace('\n','')

        # build a dataframe
        df = pd.DataFrame({'sentence':out})

        #sentimental analysis
        sid = SentimentIntensityAnalyzer()
        neg,neu,pos = [],[],[]
        senti = []
        for i in out:
            score = sid.polarity_scores(i)
            neg.append(score['neg'])
            neu.append(score['neu'])
            pos.append(score['pos'])
            if score['compound']<=-0.05: senti.append('Negative')
            elif score['compound']>=0.05: senti.append('Positive')
            else: senti.append('Neutral')

        com = []
        score = sid.polarity_scores(text)
        if score['compound']<=-0.05: com.append('Negative')
        elif score['compound']>=0.05: com.append('Positive')
        else: com.append('Neutral')
        com += [score['neg'],score['neu'],score['pos']]

        t,word = Topic_modeling(df)
        return render_template('success.html', ti = title, o = out,s = senti, n = neg, nu = neu, p = pos, la = com, to = t, wo = word, zip=zip)

if __name__ == '__main__':
    app.run(debug=True)
    app.jinja_env.filters['zip'] = zip
