# Necessary Libraries Importing
import joblib
from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

##############################################################################################################################################################################################

app = Flask(__name__)

#################################################################################################################################################################################################

# Load the model using joblib and pickle
model = pickle.load(open('cancer.pkl', 'rb'))
model1 = joblib.load(open('heart.pkl', 'rb'))
model2 = joblib.load('liver.pkl')
model3 = joblib.load(open('diabetes.pkl', 'rb'))
model4 = joblib.load(open('kidney.pkl', 'rb'))

###############################################################################################################################################################################################

# Html File routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/index')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/help')
def help():
    return render_template("help.html")

@app.route('/terms')
def terms():
    return render_template("tc.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/disindex")
def disindex():
    return render_template("disindex.html")

@app.route("/cancer")
def cancer():
    return render_template("cancer.html")

@app.route("/heart")
def heart():
    return render_template("heart.html")

@app.route("/kidney")
def kidney():
    return render_template("kidney.html")

@app.route("/liver")
def liver():
    return render_template("liver.html")

@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")

##############################################################################################################################################################################################

# Kidney Disease prediction route
@app.route('/predictkidney', methods=['POST']) 
def predictkidney():
    if request.method == 'POST':
        # Extract user input from the form
        age = int(request.form['age'])
        blood_pressure = int(request.form['blood_pressure'])
        albumin = float(request.form['albumin'])
        sugar = int(request.form['sugar'])
        pus_cell = int(request.form['pus_cell'])
        blood_urea = int(request.form['blood_urea'])
        serum_creatinine = float(request.form['serum_creatinine'])
        haemoglobin = float(request.form['haemoglobin'])
        white_blood_cell_count = int(request.form['white_blood_cell_count'])
        red_blood_cell_count = float(request.form['red_blood_cell_count'])

        # Create a DataFrame with all the user input
        user_data = pd.DataFrame({
            'Age': [age],
            'Blood Pressure': [blood_pressure],
            'Albumin': [albumin],
            'Blood Sugar Level': [sugar],
            'Pus Cell Count': [pus_cell],
            'Blood Urea': [blood_urea],
            'Serum Creatinine': [serum_creatinine],
            'Haemoglobin': [haemoglobin],
            'White Blood Cell Count': [white_blood_cell_count],
            'Red Blood Cell Count': [red_blood_cell_count],
        })
        
        # Perform kidney prediction using your trained model
        output = model4.predict(user_data)

        # Generate a Pandas report
        prediction_report = generate_pandas_report(user_data, output)

        # Pass the prediction, report, and user data to the template
        return render_template('kidney_result.html', prediction=output, prediction_report=prediction_report, user_data=user_data)


#############################################################################################################################################################################################

# Liver Disease prediction route
@app.route('/predictliver', methods=['POST']) 
def predictliver():
    if request.method == 'POST':
        # Extract user input from the form
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        total_bilirubin = float(request.form['total_bilirubin'])
        direct_bilirubin = float(request.form['direct_bilirubin'])
        alkaline_phosphotase = int(request.form['alkaline_phosphotase'])
        alamine_aminotransferase = int(request.form['alamine_aminotransferase'])
        aspartate_aminotransferase = int(request.form['aspartate_aminotransferase'])
        total_protiens = float(request.form['total_protiens'])
        albumin = float(request.form['albumin'])
        albumin_and_globulin_ratio = float(request.form['albumin_and_globulin_ratio']) 

        # Create a DataFrame with the user input
        user_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Total Bilirubin': [total_bilirubin],
            'Direct Bilirubin': [direct_bilirubin],
            'Alkaline Phosphotase': [alkaline_phosphotase],
            'Alamine Aminotransferase': [alamine_aminotransferase],
            'Aspartate Aminotransferase': [aspartate_aminotransferase],
            'Total Proteins': [total_protiens],
            'Albumin': [albumin],
            'Albumin And Globulin Ratio': [albumin_and_globulin_ratio],
        })
        
        # Perform liver prediction using your trained model
        output = model2.predict(user_data)

        # Generate a Pandas report
        prediction_report = generate_pandas_report(user_data, output)

        # Pass the prediction, report, and user data to the template
        return render_template('liver_result.html', prediction=output, prediction_report=prediction_report, user_data=user_data)

##############################################################################################################################################################################################

# Cancer prediction route
@app.route('/predict', methods=['POST']) 
def predict():
    if request.method == 'POST':
        # Extract user input from the form
        clump_thickness = int(request.form['clump_thickness'])
        uniform_cell_size = int(request.form['uniform_cell_size'])
        uniform_cell_shape = int(request.form['uniform_cell_shape'])
        marginal_adhesion = int(request.form['marginal_adhesion'])
        single_epithelial_size = int(request.form['single_epithelial_size'])
        bare_nuclei = int(request.form['bare_nuclei'])
        bland_chromatin = int(request.form['bland_chromatin'])
        normal_nucleoli = int(request.form['normal_nucleoli'])
        mitoses = int(request.form['mitoses'])

        # Create a DataFrame with the user input
        user_data = pd.DataFrame({
            'Clump Thickness': [clump_thickness],
            'Uniform Cell size': [uniform_cell_size],
            'Uniform Cell shape': [uniform_cell_shape],
            'Marginal Adhesion': [marginal_adhesion],
            'Single Epithelial Cell Size': [single_epithelial_size],
            'Bare Nuclei': [bare_nuclei],
            'Bland Chromatin': [bland_chromatin],
            'Normal Nucleoli': [normal_nucleoli],
            'Mitoses': [mitoses],
        })
        
        # Perform cancer prediction using your trained model
        output = model.predict(user_data)

        # Generate a Pandas report
        prediction_report = generate_pandas_report(user_data, output)

        # Pass the prediction, report, and user data to the template
        return render_template('cancer_result.html', prediction=output, prediction_report=prediction_report, user_data=user_data)

###############################################################################################################################################################################################

# Heart Disease prediction route
@app.route('/predictheart', methods=['POST']) 
def predictheart():
    if request.method == 'POST':
        # Extract user input from the form
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        CP = int(request.form['CP'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        FBS = int(request.form['FBS'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        EXANG = int(request.form['EXANG'])
        oldpeak = float(request.form['oldpeak'])
        SLOPE = int(request.form['SLOPE'])
        CA = int(request.form['CA'])
        THAL = int(request.form['THAL'])

        # Create a DataFrame with the user input
        user_data = pd.DataFrame({
            'Age in Years': [age],
            'Sex': [sex],
            'CP': [CP],
            'Trest Bps': [trestbps],
            'Cholesterol': [chol],
            'FBS': [FBS],
            'RESTECG': [restecg],
            'Thalach': [thalach],
            'EXANG': [EXANG],
            'Old Peak': [oldpeak],
            'SLOPE': [SLOPE],
            'CA': [CA],
            'THAL': [THAL],
        })
        
        # Perform heart disease prediction using your trained model
        output = model1.predict(user_data)

        # Generate a Pandas report
        prediction_report = generate_pandas_report(user_data, output)

        # Pass the prediction, report, and user data to the template
        return render_template('heart_result.html', prediction=output, prediction_report=prediction_report, user_data=user_data)

##############################################################################################################################################################################################

# Diabetes prediction route
@app.route('/predictt', methods=['POST'])
def predictt():
    if request.method == 'POST':
        # Extract user input from the form
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])

        # Create a DataFrame with the user input
        user_data = pd.DataFrame({
            'Pregnancies': [preg],
            'Glucose': [glucose],
            'BloodPressure': [bp],
            'SkinThickness': [st],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DPF': [dpf],
            'Age': [age]
        })

        # Perform diabetes prediction using your trained model
        output = model3.predict(user_data)

        # Generate a Pandas report
        prediction_report = generate_pandas_report(user_data, output)

        # Pass the prediction, report, and user data to the template
        return render_template('diab_result.html', prediction=output, prediction_report=prediction_report, user_data=user_data)

###############################################################################################################################################################################################

def generate_pandas_report(user_data, prediction):
    # Your actual report generation logic
    # This is a placeholder, replace it with the actual logic based on your requirements
    report_html = f"<p>User Data: {user_data.to_html()}</p><p>Prediction: {prediction}</p>"
    return report_html 

################################################################################################################################################################################################

if __name__ == '__main__':
    app.run(debug=True)


