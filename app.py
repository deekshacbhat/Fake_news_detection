from flask import Flask, render_template, request
from api import main_prediction_output  # your existing API logic

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news_text']              # get text from form
    langs = request.form.getlist('lang')               # optional languages list
    try:
        output_text = main_prediction_output(news_text, langs)
    except TypeError:
        # fallback if your function only accepts 1 argument
        output_text = main_prediction_output(news_text)
    return render_template('index.html', output=output_text, news_text=news_text)

if __name__ == '__main__':
    app.run(debug=True)
