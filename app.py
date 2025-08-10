from flask import Flask, request, render_template, url_for
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Initialize Gemini API if key is available
gemini_available = False
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        gemini_available = True
        logger.info("Gemini API configured successfully")
    except Exception as e:
        logger.warning(f"Gemini API not available: {str(e)}")
        gemini_available = False
else:
    logger.warning("Gemini API key not found in .env file")

# Load model components
try:
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    selected_features = joblib.load('selected_features.pkl')
    logger.info(f"Model loaded successfully. Selected features: {selected_features}")
except Exception as e:
    logger.error(f"Error loading model components: {str(e)}")
    logger.error("Please run 'python train.py' first to train the model")
    raise

# Load original dataset for reference values
try:
    data = pd.read_csv('diabetes.csv')
    logger.info("Dataset loaded for reference values")
except Exception as e:
    logger.error(f"Error loading dataset: {str(e)}")
    raise

def get_gemini_recommendations(probability, glucose, bmi, age):
    """Get AI-powered recommendations using Gemini API"""
    if not gemini_available:
        return None
    
    risk_level = "High" if probability > 0.7 else "Moderate" if probability > 0.4 else "Low"
    
    prompt = f"""
    Generate 4 concise, actionable lifestyle recommendations for a patient with {risk_level} diabetes risk.
    
    Patient Profile:
    - Diabetes Risk Probability: {probability:.1%}
    - Glucose Level: {glucose:.1f} mg/dL
    - BMI: {bmi:.1f}
    - Age: {age} years
    
    Provide recommendations for:
    1. Diet/Nutrition
    2. Physical Activity
    3. Health Monitoring
    4. Lifestyle/Sleep
    
    Keep each recommendation to 1-2 sentences and make them specific and actionable.
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        recommendations = [line.strip() for line in response.text.split('\n') if line.strip()]
        # Take first 4 non-empty recommendations
        return recommendations[:4] if len(recommendations) >= 4 else recommendations
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        return None

def get_static_recommendations(probability):
    """Fallback static recommendations"""
    if probability > 0.7:
        return [
            "Consult a healthcare provider immediately for comprehensive diabetes screening and management plan.",
            "Adopt a low-glycemic diet with complex carbohydrates, lean proteins, and healthy fats.",
            "Engage in at least 150 minutes of moderate aerobic activity per week, plus strength training.",
            "Monitor blood glucose levels regularly and maintain a consistent sleep schedule of 7-8 hours."
        ]
    elif probability > 0.4:
        return [
            "Schedule a check-up with your doctor to discuss your diabetes risk factors.",
            "Incorporate 30 minutes of daily physical activity like brisk walking or swimming.",
            "Reduce sugar and processed food intake; focus on whole grains and vegetables.",
            "Stay hydrated, manage stress levels, and aim for consistent sleep patterns."
        ]
    else:
        return [
            "Maintain a balanced diet rich in fruits, vegetables, whole grains, and lean proteins.",
            "Stay active with regular exercise, aiming for at least 150 minutes per week.",
            "Monitor your weight and maintain a healthy BMI between 18.5-24.9.",
            "Get regular health check-ups and maintain good sleep hygiene."
        ]

def process_input_features(form_data):
    """Process form input and create feature vector"""
    try:
        # Extract basic features
        pregnancies = float(form_data['pregnancies'])
        glucose = float(form_data['glucose'])
        blood_pressure = float(form_data['blood_pressure'])
        skin_thickness = float(form_data['skin_thickness'])
        insulin = float(form_data['insulin'])
        bmi = float(form_data['bmi'])
        diabetes_pedigree = float(form_data['diabetes_pedigree'])
        age = float(form_data['age'])

        # Handle zero values with median replacement
        if glucose == 0:
            glucose = data['Glucose'].median()
        if blood_pressure == 0:
            blood_pressure = data['BloodPressure'].median()
        if skin_thickness == 0:
            skin_thickness = data['SkinThickness'].median()
        if insulin == 0:
            insulin = data['Insulin'].median()
        if bmi == 0:
            bmi = data['BMI'].median()

        # Create ALL features in the exact same order as training
        all_features = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': diabetes_pedigree,
            'Age': age,
            'Glucose_BMI': glucose * bmi,
            'Age_DiabetesPedigree': age * diabetes_pedigree,
            'Insulin_Glucose_Ratio': insulin / (glucose + 1e-6),
            'BMI_Category': float(0 if bmi < 18.5 else 1 if bmi < 25 else 2 if bmi < 30 else 3),
            'Glucose_Age': glucose * age
        }

        # Create DataFrame with ALL features first (same as training)
        feature_order = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
                        'BMI', 'DiabetesPedigreeFunction', 'Age', 'Glucose_BMI', 'Age_DiabetesPedigree', 
                        'Insulin_Glucose_Ratio', 'BMI_Category', 'Glucose_Age']
        
        full_data = pd.DataFrame([all_features], columns=feature_order)
        
        # Scale the full feature set
        full_data_scaled = scaler.transform(full_data)
        full_data_scaled = pd.DataFrame(full_data_scaled, columns=feature_order)
        
        # Now select only the features chosen by RFE
        input_data = full_data_scaled[selected_features]
        
        logger.info(f"Processed features: {list(input_data.columns)}")
        logger.info(f"Expected features: {selected_features}")
        
        return input_data, glucose, bmi, age
        
    except Exception as e:
        logger.error(f"Error processing input features: {str(e)}")
        raise ValueError(f"Invalid input data: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            logger.info("Processing prediction request")
            
            # Process input features
            input_data, glucose, bmi, age = process_input_features(request.form)
            
            # Make prediction (input_data is already scaled and feature-selected)
            probability = model.predict_proba(input_data)[0][1]
            prediction = model.predict(input_data)[0]
            
            # Determine risk level
            if probability > 0.7:
                risk_level = "High Risk"
                risk_class = "text-red-600"
            elif probability > 0.4:
                risk_level = "Moderate Risk"
                risk_class = "text-yellow-600"
            else:
                risk_level = "Low Risk"
                risk_class = "text-green-600"
            
            prediction_text = f"{risk_level} ({probability:.1%} probability)"
            
            # Get recommendations
            recommendations = get_gemini_recommendations(probability, glucose, bmi, age)
            if not recommendations:
                recommendations = get_static_recommendations(probability)
            
            logger.info(f"Prediction successful: {prediction_text}")
            
            return render_template('index.html',
                                 prediction=prediction_text,
                                 risk_class=risk_class,
                                 recommendations=recommendations,
                                 show_results=True,
                                 probability=probability)
                                 
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            error_message = f"Prediction Error: {str(e)}"
            return render_template('index.html',
                                 prediction=error_message,
                                 risk_class="text-red-600",
                                 recommendations=None,
                                 show_results=False)
    
    return render_template('index.html',
                         prediction=None,
                         recommendations=None,
                         show_results=False)

@app.route('/static/<filename>')
def static_files(filename):
    """Serve static files"""
    return app.send_static_file(filename)

if __name__ == '__main__':
    print("Starting Diabetes Risk Prediction App...")
    print(f"Gemini AI recommendations: {'Enabled' if gemini_available else 'Disabled (using static recommendations)'}")
    print("Navigate to http://127.0.0.1:5000 in your browser")
    
    try:
        app.run(debug=True, host='127.0.0.1', port=5000)
    except Exception as e:
        logger.error(f"Error starting Flask app: {str(e)}")