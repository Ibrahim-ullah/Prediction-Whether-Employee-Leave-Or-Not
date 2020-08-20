import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction

    if output == 1:
    	answer = "The employee will leave you"
    elif output == 0:	
    	answer = "The employee won't leave you"	

    return render_template('index.html', prediction_text=answer)


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)


#if __name__ == "__main__":
#    app.run(debug=True)


