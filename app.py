from flask import Flask,render_template,request
import pickle
import numpy as np
app = Flask('__name__')
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=["POST"])
def predict():
    feature=[float(x) for x in request.form.values()]
    feature_final=np.array(feature).reshape(-1,1)
    prediction=model.predict(feature_final)
    return render_template('index.html',prediction_text=' Sınav Sonucu Tahmini = {}'.format(int(prediction)))

if(__name__=='__main__'):
    app.run(debug=True)


