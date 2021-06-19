#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import random

# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image




from flask import Flask, redirect, url_for, request, render_template,jsonify
from werkzeug.utils import secure_filename

import pickle





app = Flask(__name__)


#Loading the models
model = pickle.load(open('breastCancer.pkl', 'rb'))

model_cardiovascular=pickle.load(open('echocardiogramResults.pkl', 'rb'))
model_cardiovascular_scalar=pickle.load(open('echocardiogramScaler.pkl', 'rb'))

model_covid=load_model('/app/covid_model_resnet50_b32_e5_acc92.23.h5')

model_brain=load_model('brain_model_vgg19_b16_ep_10_acc_93.33.h5')

model_malaria=load_model("MalariaAcc92-71E5B32.h5")

model_urineDiagnosis=load_model("urineDiagnosis_modela100t0.6bs15.h5")
model_urineDiagnosis_scalar=pickle.load(open("urinaryDisorderScaler.pkl","rb"))

model_heartFailure=pickle.load(open('heartFailureAcc87r90.pkl','rb'))

model_hypertension=load_model("HypertensionAcc70r70.h5")
model_hypertension_scalar=pickle.load(open("HypertensionScaler.pkl","rb"))

model_stroke=pickle.load(open('StrokeAcc1r1.pkl',"rb"))

model_diabetes=load_model("DiabetesAcc76rec0-85-1-71.h5")



model_plantarFascitis=pickle.load(open('PlantarFascitisAcc94r91.pkl',"rb"))
model_plantarFascitis_scalar=pickle.load(open('PlantarFascitisScaler.pkl',"rb"))

model_fertility=pickle.load(open('fertilityResultsA94R1-100R0-90.pkl',"rb"))
model_fertility_scalar=pickle.load(open('fertilityScaler.pkl',"rb"))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index.html')
def home2():
    return render_template('index.html')


#Breast Cancer Model
@app.route('/breastCancerPrediction.html', methods=['POST','GET'])
def breast_cancer_predict():

    if request.method == 'GET':
        return render_template('breastCancerPrediction.html')
    else:
        try:
            int_features = [x for x in request.form.values()]
            final_features = [np.array(int_features)]
            prediction = model.predict(final_features)
            if(prediction[0]==1):
                return render_template('Result.html', prediction_text=f'Unfortunately, your tumour has been diagnosed as malignant with an accuracy of 96% and sensitivity of 97%')
            else:
                return render_template('Result.html', prediction_text=f'Fortunately, your tumour has been diagnosed as benign with an accuracy of 96% and sensitivity of 96%')
        except:
            print("Please enter the details first !!", "danger")      
            
#Echocardiogram Model
@app.route("/EchocardiogramAnalysis.html", methods=['POST','GET'])
def echocardiogramAnalysis():

    if request.method == 'GET':
        return render_template('EchocardiogramAnalysis.html')
    else:
        try:
            int_features = [x for x in request.form.values()]
           
            if(int_features[1]=="no"):
                int_features[1]=0
            else:
                int_features[1]=1
            
            features=[np.array(int_features)]
            
            df = pd.DataFrame(features, columns = ['Age at heart attack','pericardial-effusion','fractional-shortening', 'epss', 'lvdd',
       'wall-motion-index'])
            columns_to_scale = ['Age at heart attack','epss', 'lvdd','wall-motion-index']
            df[columns_to_scale]=model_cardiovascular_scalar.transform(df[columns_to_scale])
            prediction = model_cardiovascular.predict(df)
            print(prediction[0])
                
            if(prediction[0]==0):
                return render_template('Result.html', prediction_text=f'With an accuracy of 94% and sensitivity of 100%, unfortunately, you need immediate treatment as you are less likely to survive for atleast 1 year')
            else:
                return render_template('Result.html', prediction_text=f'With an accuracy of 94% and sensitivity of 87%, you are likely to survive for atleast 1 year, however there are 13% chances you have been diagnosed incorrectly, please do further checkups')
        except:
            print("Please enter the details first !!", "danger")      


def covid_predict(img_path, model):
    from tensorflow.keras.applications.resnet import preprocess_input 
    img = image.load_img(img_path, target_size=(224, 224))       
    # Preprocessing the image
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    img_data=preprocess_input(x)  
    a=np.argmax(model.predict(img_data), axis=1)
    return a
  
def brainHemmorhage_predict(img_path, model):
    from tensorflow.keras.applications.vgg19 import preprocess_input
    img = image.load_img(img_path, target_size=(224, 224))       
    # Preprocessing the image
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    img_data=preprocess_input(x)  
    a=np.argmax(model.predict(img_data), axis=1)
    return a

def malaria_predict(img_path, model):
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    img = image.load_img(img_path, target_size=(299, 299))       
    # Preprocessing the image
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    img_data=preprocess_input(x)  
    a=np.argmax(model.predict(img_data), axis=1)
    return a


#COVID Detection
@app.route('/COVIDDetection.html', methods=['POST','GET'])
def covid_detect():
    return render_template('COVIDDetection.html')

@app.route('/predictCOVID.html', methods=['GET', 'POST'])
def covidCheck():
     if request.method == 'GET':
        return render_template('predictCOVID.html')
     else:
         # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname("__file__")
        file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        
        a = covid_predict(file_path, model_covid)
        if(a==0):
            preds="With an accuracy of 92.23% you have been diagnosed with COVID"
        elif(a==1):
            preds="With an accuracy of 92.23% your lungs are detected normal"
        else:
            preds="With an accuracy of 92.23% you have been diagnosed with Viral Pneumonia"
        return preds

    
#Hemmorrhage Detection
@app.route("/BrainHemorrhageDetection.html", methods=['POST','GET'])
def brainblood_detect():
    return render_template('BrainHemorrhageDetection.html')

@app.route('/predictHemmorhage.html', methods=['GET', 'POST'])
def hemmorhageCheck():
     if request.method == 'GET':
        return render_template('predictHemmorhage.html')
     else:
         # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname("__file__")
        file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        
        a = brainHemmorhage_predict(file_path, model_brain)
        if(a==0):
            preds="With an accuracy of 93.33% you do not have Brain Hemmorhage, but please go for further tests."
        else:
             preds="With an accuracy of 93.33% you are diagnosed with Brain Hemmorhage"
        return preds
    
#Urinary Disorder Model
@app.route('/urineDisorderPrediction.html', methods=['POST','GET'])
def urine_disorder_predict():

    if request.method == 'GET':
        return render_template('urineDisorderPrediction.html')
    else:
        try:
            int_features = [x for x in request.form.values()]
            
            features=[np.array(int_features)]
            
            
            df = pd.DataFrame(features, columns = ["Temperature",'Occurrence of nausea_yes', 'Lumbar pain_yes',
       'Urine pushing_yes', 'Micturition pains_yes',
       'Burning of urethra, itch, swelling of urethra outlet_yes'])
            
            df=df.astype(int)
            
            columns_to_scale = ["Temperature"]
            df[columns_to_scale]=model_urineDiagnosis_scalar.transform(df[columns_to_scale])
            prediction = model_urineDiagnosis.predict(np.asarray(df.to_numpy()))
            
            if (prediction[0][0]>0.5 and prediction[0][1]>0.5):
                return render_template('Result.html', prediction_text=f"Inflammation of urinary bladder as well as nephritis of renal pelvis origin detected with an accuracy of 100%")
            elif (prediction[0][0]>0.5):
                return render_template('Result.html', prediction_text=f"Inflammation of urinary bladder detected with an accuracy of 100%")
            elif(prediction[0][1]>0.5):
                return render_template('Result.html', prediction_text=f"Nephritis of renal pelvis origin detected detected with an accuracy of 100%")
            else:
                return render_template('Result.html', prediction_text=f"No issue detected with an accuracy of 100%")
        except:
            print("Please enter the details first !!", "danger")     

#Heart Failure Model
@app.route('/heartFailure.html', methods=['POST','GET'])
def heart_failure_predict():
    if request.method == 'GET':
        return render_template('heartFailure.html')
    else:
        try:
           
            int_features = [x for x in request.form.values()]
            final_features = [np.array(int_features)]
            df = pd.DataFrame(final_features, columns = ['age','creatinine_phosphokinase','ejection_fraction','high_blood_pressure',
 'platelets','serum_creatinine','serum_sodium','smoking','time'])      
            df=df.astype(float)
            prediction = model_heartFailure.predict(df)
            print("The prediction is",prediction)
            if(prediction[0]==1):
                return render_template('Result.html', prediction_text=f'Unfortunately,the patient is likely to suffer a heart failure in the future time with an accuracy of 87% and sensitivity of 90%')
            else:
                return render_template('Result.html', prediction_text=f'Fortunately, the patient is not likely to suffer a heart failure in the future time with an accuracy of 87% and sensitivity of 85%')
        except:
            print("Please enter the details first !!", "danger")   
            
#Hypertension Model
@app.route('/hypertensionDetection.html', methods=['POST','GET'])
def hypertension_predict():

    if request.method == 'GET':
        return render_template('hypertensionDetection.html')
    else:
        try:
            int_features = [x for x in request.form.values()]
            
            
            features=[np.array(int_features)]
            
           
            df = pd.DataFrame(features, columns = ['GENDER', 'AGERANGE', 'RACE', 'BMIRANGE','SMOKE',
       'DIABETES'])
          
            
         
           
            
            df=model_hypertension_scalar.transform(df)
           
            prediction = model_hypertension.predict(np.asarray(df))
            
            if (prediction[0]==0):
                return render_template('Result.html', prediction_text=f"Congratulations, with an accuracy of 70% and sensitivity of 70% you do not suffer from hypertension")
            else:
                return render_template('Result.html', prediction_text=f"Unfortunately, with an accuracy of 70% and sensitivity of 70% you are suffering from hypertension")
        except:
            print("Please enter the details first !!", "danger")     
            
            
#Stroke Prediction Model
@app.route('/strokePrediction.html', methods=['POST','GET'])
def stroke_predict():

    if request.method == 'GET':
        return render_template('strokePrediction.html')
    else:
        try:
           
            int_features = [x for x in request.form.values()]
            final_features = [np.array(int_features)]

            df = pd.DataFrame(final_features, columns = ['age', 'hypertension', 'heart_disease', 'work_type',
       'avg_glucose_level', 'bmi', 'gender_Male', 'ever_married_Yes',
       'Residence_type_Urban', 'smoking_status'])   
           
            df=df.astype(float)
            prediction = model_stroke.predict(df)
       
            if(prediction[0]==1):
                return render_template('Result.html', prediction_text=f'Unfortunately,the patient is likely to encounter a stroke in the future time with an accuracy and sensitivity of 100%')
            else:
                return render_template('Result.html', prediction_text=f'Fortunately, the patient is not likely to encounter a stroke in the future time with an accuracy and sensitivity of 100%')
        except:
            print("Please enter the details first !!", "danger") 
            
#Diabetes Model
@app.route('/diabetesDetection.html', methods=['POST','GET'])
def diabetes_predict():

    if request.method == 'GET':
        return render_template('diabetesDetection.html')
    else:
        try:
            int_features = [x for x in request.form.values()]
            
            
            features=[np.array(int_features)]
            
            print("The features are",features)
            df = pd.DataFrame(features, columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])
            df=df.astype(float)
            print("The data is",df)
            

            prediction = model_diabetes.predict(np.asarray(df))
            
            if (prediction[0][0]==0):
                return render_template('Result.html', prediction_text=f"Congratulations, with an accuracy of 76% and sensitivity of 85% you are not diagnosed with diabetes")
            else:
                return render_template('Result.html', prediction_text=f"Unfortunately, with an accuracy of 76% and sensitivity of 71% you are diagnosed with diabetes")
        except:
            print("Please enter the details first !!", "danger")  
            
            
#Plantar Fascitis model
@app.route("/plantarFascitisDetection.html", methods=['POST','GET'])
def plantarFascitis():

    if request.method == 'GET':
        return render_template('plantarFascitisDetection.html')
    else:
        try:
            int_features = [x for x in request.form.values()]
           
            features=[np.array(int_features)]
            
            df = pd.DataFrame(features, columns =["age","weight",'sex','hoursRunInAWeek','allignmentOfMidFoot', 'diabetes'])
            columns_to_scale = ["age","weight",'hoursRunInAWeek']
            df[columns_to_scale]=model_plantarFascitis_scalar.transform(df[columns_to_scale])
            prediction = model_plantarFascitis.predict(df)
           
                
            if(prediction[0]==0):
                return render_template('Result.html', prediction_text=f'With an accuracy of 94% and sensitivity of 95%, congratulations, you are not diagnosed with Plantar Fascitis')
            else:
                return render_template('Result.html', prediction_text=f'With an accuracy of 94% and sensitivity of 91%, unfortunately, you are diagnosed with Plantar Fascitis')
        except:
            print("Please enter the details first !!", "danger")      

#Fertility diagnosis
@app.route('/maleFertilityDetection.html', methods=['POST','GET'])
def fertility_predict():

    if request.method == 'GET':
        return render_template('maleFertilityDetection.html')
    else:
        try:
            int_features = [x for x in request.form.values()]
            
            
            features=[np.array(int_features)]
           
            df = pd.DataFrame(features, columns =['Season', 'Childish diseases', 'Accident or serious trauma','Surgical intervention', 'High fevers in the last year','Frequency of alcohol consumption', 'Smoking habit'])
            
           
            
            df=model_fertility_scalar.transform(df)
            
           
            prediction = model_fertility.predict(np.asarray(df))
          
                                       
            if (prediction[0]==0):
                return render_template('Result.html', prediction_text=f"Congratulations, with an accuracy of 94% and sensitivity of 90% your semen sample is normal")
            else:
                return render_template('Result.html', prediction_text=f"Unfortunately, with an accuracy of 94% and sensitivity of 100% your semen sample is not normal citing a possibiliy of asthenozoospermia or oligozoospermia")
        except:
            print("Please enter the details first !!", "danger")  
            
#Malaria Detection
@app.route("/Malaria Detection.html", methods=['POST','GET'])
def malaria_detect():
    return render_template('Malaria Detection.html')

@app.route('/predictMalaria.html', methods=['GET', 'POST'])
def malariaCheck():
     if request.method == 'GET':
        return render_template('predictMalaria.html')
     else:
         # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname("__file__")
        file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        
        a = malaria_predict(file_path, model_malaria)
        if(a==0):
            preds="With an accuracy of 92.71% you are diagnosed with malaria."
        else:
             preds="With an accuracy of 92.71% you are not diagnosed with malaria"
        return preds
        
if __name__ == '__main__':
    app.run(debug=True)

