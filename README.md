# Diabetes-Risk-Prediction-Lifestyle-Recommendation
Predict diabetes likelihood and provide lifestyle recommendations based on medical history, BMI, glucose levels, and activity.

## 🌟 Features

- **Advanced ML Pipeline**: Ensemble of RandomForest, XGBoost, and LightGBM
- **Smart Data Processing**: Handles missing values and outliers automatically
- **Feature Engineering**: Creates interaction features for better predictions
- **AI Recommendations**: Personalized advice powered by Google Gemini AI
- **Interactive Web Interface**: Beautiful, responsive design with tooltips
- **Class Imbalance Handling**: SMOTE oversampling for better accuracy
- **Feature Selection**: RFE for optimal feature subset

## 📊 Model Performance

- **Ensemble Accuracy**: ~82-85% on test data
- **Feature Selection**: Automatically selects 8 most important features
- **Class Balance**: SMOTE handles dataset imbalance
- **Robust Preprocessing**: Median imputation and outlier capping

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Diabetes dataset (diabetes.csv)
- Google Gemini API key (optional but recommended)

### 1. Setup Project
```bash
# Clone or download all files to a directory
# Ensure you have diabetes.csv in the project folder
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file with your Gemini API key:
```
GEMINI_API_KEY=your_actual_api_key_here
```

Get your API key from: https://makersuite.google.com/app/apikey

### 4. Run Setup (Optional)
```bash
python setup.py
```

### 5. Quick Start
```bash
python run.py
```

Or manually:
```bash
# Train the model
python train.py

# Start the web app
python app.py
```

### 6. Access Application
Open your browser and go to: `http://127.0.0.1:5000`

## 📁 Project Structure

```
diabetes-prediction/
├── train.py              # Model training script
├── app.py                # Flask web application
├── requirements.txt      # Python dependencies
├── setup.py             # Setup verification script
├── run.py               # Quick start script
├── .env                 # Environment variables
├── diabetes.csv         # Dataset (you need to provide this)
├── templates/
│   └── index.html       # Web interface template
├── static/
│   └── feature_importance.png  # Generated plot
├── diabetes_model.pkl   # Trained model (generated)
├── scaler.pkl          # Feature scaler (generated)
└── selected_features.pkl # Selected features (generated)
```

## 🔬 Technical Details

### Machine Learning Pipeline
1. **Data Preprocessing**:
   - Median imputation for zero values
   - IQR-based outlier capping
   - StandardScaler normalization

2. **Feature Engineering**:
   - Glucose_BMI interaction
   - Age_DiabetesPedigree interaction
   - Insulin_Glucose_Ratio
   - BMI_Category binning
   - Glucose_Age interaction

3. **Model Training**:
   - RFE feature selection (8 features)
   - SMOTE oversampling
   - Grid search hyperparameter tuning
   - Soft voting ensemble

4. **Models Used**:
   - Random Forest with class weights
   - XGBoost with scale_pos_weight
   - LightGBM with class weights

### Risk Assessment
- **Low Risk**: <40% probability (Green)
- **Moderate Risk**: 40-70% probability (Yellow)
- **High Risk**: >70% probability (Red)

## 📝 Input Features

| Feature | Range | Description |
|---------|-------|-------------|
| Pregnancies | 0-17 | Number of pregnancies |
| Glucose | 0-200 mg/dL | Blood glucose concentration |
| Blood Pressure | 0-122 mm Hg | Diastolic blood pressure |
| Skin Thickness | 0-99 mm | Triceps skin fold thickness |
| Insulin | 0-846 mu U/ml | 2-hour serum insulin |
| BMI | 0-67.1 | Body mass index |
| Diabetes Pedigree | 0.078-2.42 | Genetic risk function |
| Age | 21-81 years | Age in years |

*Note: Enter 0 for unknown values - the system will use median estimates*

## 🤖 AI Recommendations

The system provides personalized lifestyle recommendations based on:
- Risk probability level
- Individual health metrics
- Age and BMI considerations
- Google Gemini AI analysis (if configured)

## 🛠️ Troubleshooting

### Common Issues

1. **Module Import Errors**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Dataset Not Found**:
   - Download diabetes.csv from Kaggle: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
   - Place in project root directory

3. **Gemini API Errors**:
   - Verify API key in .env file
   - Check API quota limits
   - System works with static recommendations if Gemini fails

4. **Model Training Fails**:
   - Ensure sufficient memory (>2GB recommended)
   - Check dataset format and completeness
   - Verify all dependencies are installed

5. **Web App Won't Start**:
   - Check if port 5000 is available
   - Ensure model files exist (run train.py first)
   - Check Flask installation

### Error Logs
The application provides detailed logging. Check console output for specific error messages.

## 📊 Expected Performance

- **Training Time**: 2-5 minutes on modern hardware
- **Prediction Time**: <1 second per prediction
- **Memory Usage**: ~500MB during training
- **Accuracy**: 82-85% on test data

## 🔒 Privacy & Security

- No data is stored permanently
- All processing happens locally
- Gemini API calls only send aggregated risk metrics
- No personal health data is transmitted

## 🤝 Contributing

Feel free to enhance the system by:
- Adding more sophisticated models
- Improving the web interface
- Adding more comprehensive recommendations
- Implementing additional validation

## 📄 License

This project is for educational and research purposes. Please consult healthcare professionals for medical advice.

## 🙏 Acknowledgments

- Pima Indians Diabetes Dataset from UCI ML Repository
- Google Gemini AI for recommendation generation
- Scikit-learn, XGBoost, and LightGBM communities
