from flask import Flask,request,render_template
from src.pipeline.predict_pipeline import Custom_Data,Predict_Pipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data= Custom_Data(
            age = float(request.form.get('age')),
            thalach = float(request.form.get('thalach')),
            trestbps = float(request.form.get('trestbps')),
            chol =  float(request.form.get('chol')),
            oldpeak = float(request.form.get('oldpeak')),
            sex = request.form.get('sex'),
            cp = float(request.form.get('cp')),
            fbs = request.form.get('fbs'),
            restecg = float(request.form.get('restecg')),
            
            exang = request.form.get('exang'),
            
            slope = request.form.get('slope'),
            ca = request.form.get('ca'),
            thal = request.form.get('thal')
        )
       
        pred_df = Custom_Data.get_custom_data(data)
        print(pred_df)
        print("Before Prediction")
        print(pred_df.dtypes)
        predict_pipeline = Predict_Pipeline()
        results = predict_pipeline.predict(pred_df)
        if results == 0:
            result = "No Risk of Heart Disease"
        else:
            result = "Risk of Heart Disease"
        
        return render_template('home.html',results=result)

if __name__ == "__main__":
    app.run(host= "0.0.0.0")
