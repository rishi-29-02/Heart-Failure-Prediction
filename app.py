import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    inputs = [float(x) for x in request.form.values()]
    print(inputs)
    final_inputs = [inputs]
    prediction = model.predict(final_inputs)
    if prediction == 1:
      return render_template('index.html', prediction_text='Person passed away due to heart failure')
    else:
      return render_template('index.html', prediction_text='Person is alive.')


if __name__ == "__main__":
    app.run(debug=True)

