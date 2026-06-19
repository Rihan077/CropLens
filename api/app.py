# CropLens - Flask Backend API
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Initialize Flask app
app = Flask(
    __name__,
    template_folder='../frontend/templates',
    static_folder='../frontend/static'
)
CORS(app)

# Paths
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(BASE, 'models')
PROCESSED = os.path.join(BASE, 'data', 'processed')

# Load model and encoders
print("Loading model and encoders...")
model = joblib.load(os.path.join(MODELS, 'croplens_model.pkl'))
le_target = joblib.load(os.path.join(MODELS, 'label_encoder.pkl'))
le_crop = joblib.load(os.path.join(MODELS, 'crop_encoder.pkl'))
le_soil = joblib.load(os.path.join(MODELS, 'soil_encoder.pkl'))
le_season = joblib.load(os.path.join(MODELS, 'season_encoder.pkl'))
le_district = joblib.load(os.path.join(MODELS, 'district_encoder.pkl'))
feature_cols = joblib.load(os.path.join(MODELS, 'feature_columns.pkl'))
explainer = joblib.load(os.path.join(MODELS, 'shap_explainer.pkl'))

# Load master dataset
master_df = pd.read_csv(os.path.join(PROCESSED, 'master_dataset.csv'))

# Encode master dataset
master_df['crop_encoded'] = le_crop.transform(master_df['crop'])
master_df['soil_encoded'] = le_soil.transform(master_df['soil_type'])
master_df['season_encoded'] = le_season.transform(master_df['season'])
master_df['district_encoded'] = le_district.transform(master_df['district'])

print("✅ All models loaded successfully")

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def get_risk_details(district, crop, season):
    """Get prediction and SHAP explanation for a given input"""
    
    district = district.upper().strip()
    
    # Find matching record
    mask = (
        (master_df['district'] == district) &
        (master_df['crop'] == crop) &
        (master_df['season'].str.strip() == season)
    )
    matches = master_df[mask]
    
    if len(matches) == 0:
        return None, "No historical data found for this combination"
    
    # Take most recent record
    record = matches.iloc[-1]
    X_single = record[feature_cols].values.reshape(1, -1)
    
    # Predict
    pred_encoded = model.predict(X_single)[0]
    pred_proba = model.predict_proba(X_single)[0]
    pred_label = le_target.inverse_transform([pred_encoded])[0]
    confidence = round(float(pred_proba.max()) * 100, 1)
    risk_score = round(float(pred_proba[pred_encoded]) * 100, 1)
    
    # SHAP explanation
    shap_vals = explainer.shap_values(X_single)
    class_idx = int(pred_encoded)
    shap_for_class = shap_vals[0, :, class_idx]
    
    explanation = pd.DataFrame({
        'feature': feature_cols,
        'value': X_single[0],
        'shap_value': shap_for_class
    }).sort_values('shap_value', key=abs, ascending=False)
    
    # Top 6 reasons
    reasons = []
    for _, row in explanation.head(6).iterrows():
        direction = "increases_risk" if row['shap_value'] > 0 else "reduces_risk"
        reasons.append({
            'feature': row['feature'],
            'value': round(float(row['value']), 2),
            'direction': direction,
            'impact': round(float(abs(row['shap_value'])), 4)
        })
    
    result = {
        'district': district,
        'crop': crop,
        'season': season,
        'year': record['year'],
        'risk_level': pred_label,
        'confidence': confidence,
        'risk_score': risk_score,
        'predicted_yield': round(float(record['yield_kg_per_hectare']), 0),
        'district_avg_yield': round(float(record['district_crop_avg_yield']), 0),
        'annual_rainfall': round(float(record['annual_rainfall']), 1),
        'rainfall_deviation': round(float(record['rainfall_deviation_pct']), 1),
        'soil_type': record['soil_type'],
        'compatibility_score': round(float(record['compatibility_score']), 2),
        'drought_streak': int(record['drought_streak']),
        'reasons': reasons
    }
    
    return result, None


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/map')
def map_page():
    return render_template('map.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        district = data.get('district', '')
        crop = data.get('crop', '')
        season = data.get('season', '')
        
        if not district or not crop or not season:
            return jsonify({'error': 'Missing required fields'}), 400
        
        result, error = get_risk_details(district, crop, season)
        
        if error:
            return jsonify({'error': error}), 404
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/districts', methods=['GET'])
def get_districts():
    districts = sorted(master_df['district'].unique().tolist())
    return jsonify({'districts': districts})


@app.route('/api/crops', methods=['GET'])
def get_crops():
    crops = sorted(master_df['crop'].unique().tolist())
    return jsonify({'crops': crops})


@app.route('/api/seasons', methods=['GET'])
def get_seasons():
    seasons = sorted(master_df['season'].str.strip().unique().tolist())
    return jsonify({'seasons': seasons})


@app.route('/api/history/<district>/<crop>', methods=['GET'])
def get_history(district, crop):
    mask = (
        (master_df['district'] == district.upper()) &
        (master_df['crop'] == crop)
    )
    history = master_df[mask][
        ['year', 'yield_kg_per_hectare', 
         'annual_rainfall', 'risk_level']
    ].sort_values('year')
    
    return jsonify(history.to_dict(orient='records'))


@app.route('/api/map-data', methods=['GET'])
def get_map_data():
    # Get latest risk for each district
    latest = master_df.sort_values('year').groupby('district').last().reset_index()
    
    map_data = []
    for _, row in latest.iterrows():
        mask = (
            (master_df['district'] == row['district'])
        )
        district_data = master_df[mask]
        
        # Most common risk level for this district
        risk = district_data['risk_level'].mode()[0]
        
        map_data.append({
            'district': row['district'],
            'risk_level': risk,
            'annual_rainfall': round(float(row['annual_rainfall']), 1),
        })
    
    return jsonify(map_data)


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model': 'LightGBM',
        'version': '1.0'
    })


# ─────────────────────────────────────────
# RUN APP
# ─────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)