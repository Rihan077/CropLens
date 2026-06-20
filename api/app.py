# CropLens - Flask Backend API
import os
import joblib
import numpy as np
import pandas as pd
import requests as req
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

# District coordinates for weather API
DISTRICT_COORDS = {
    'AHMEDNAGAR': (19.0948, 74.7480),
    'AKOLA': (20.7002, 77.0082),
    'AMRAVATI': (20.9320, 77.7523),
    'AURANGABAD': (19.8762, 75.3433),
    'BEED': (18.9890, 75.7601),
    'BHANDARA': (21.1667, 79.6500),
    'BULDHANA': (20.5292, 76.1842),
    'CHANDRAPUR': (19.9615, 79.2961),
    'DHULE': (20.9013, 74.7749),
    'GADCHIROLI': (20.1809, 80.0000),
    'GONDIA': (21.4600, 80.1900),
    'HINGOLI': (19.7165, 77.1495),
    'JALGAON': (21.0077, 75.5626),
    'JALNA': (19.8347, 75.8816),
    'KOLHAPUR': (16.7050, 74.2433),
    'LATUR': (18.4088, 76.5604),
    'MUMBAI': (19.0760, 72.8777),
    'MUMBAI SUBURBAN': (19.1136, 72.8697),
    'NAGPUR': (21.1458, 79.0882),
    'NANDED': (19.1383, 77.3210),
    'NANDURBAR': (21.3653, 74.2430),
    'NASHIK': (19.9975, 73.7898),
    'OSMANABAD': (18.1860, 76.0390),
    'PALGHAR': (19.6967, 72.7650),
    'PARBHANI': (19.2704, 76.7749),
    'PUNE': (18.5204, 73.8567),
    'RAIGAD': (18.5158, 73.1298),
    'RATNAGIRI': (16.9902, 73.3120),
    'SANGLI': (16.8524, 74.5815),
    'SATARA': (17.6805, 74.0183),
    'SINDHUDURG': (16.3500, 73.8667),
    'SOLAPUR': (17.6599, 75.9064),
    'THANE': (19.2183, 72.9781),
    'WARDHA': (20.7453, 78.6022),
    'WASHIM': (20.1120, 77.1330),
    'YAVATMAL': (20.3888, 78.1204),
}

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def get_risk_details(district, crop, season):
    district = district.upper().strip()
    
    mask = (
        (master_df['district'] == district) &
        (master_df['crop'] == crop) &
        (master_df['season'].str.strip() == season)
    )
    matches = master_df[mask]
    
    if len(matches) == 0:
        return None, "No historical data found for this combination"
    
    record = matches.iloc[-1]
    X_single = record[feature_cols].values.reshape(1, -1)
    
    pred_encoded = model.predict(X_single)[0]
    pred_proba = model.predict_proba(X_single)[0]
    pred_label = le_target.inverse_transform([pred_encoded])[0]
    confidence = round(float(pred_proba.max()) * 100, 1)
    risk_score = round(float(pred_proba[pred_encoded]) * 100, 1)
    
    shap_vals = explainer.shap_values(X_single)
    class_idx = int(pred_encoded)
    shap_for_class = shap_vals[0, :, class_idx]
    
    explanation = pd.DataFrame({
        'feature': feature_cols,
        'value': X_single[0],
        'shap_value': shap_for_class
    }).sort_values('shap_value', key=abs, ascending=False)
    
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
    latest = master_df.sort_values('year').groupby('district').last().reset_index()
    
    map_data = []
    for _, row in latest.iterrows():
        mask = (master_df['district'] == row['district'])
        district_data = master_df[mask]
        risk = district_data['risk_level'].mode()[0]
        
        map_data.append({
            'district': row['district'],
            'risk_level': risk,
            'annual_rainfall': round(float(row['annual_rainfall']), 1),
        })
    
    return jsonify(map_data)


@app.route('/api/weather/<district>', methods=['GET'])
def get_weather(district):
    try:
        district = district.upper().strip()
        
        if district not in DISTRICT_COORDS:
            return jsonify({'error': 'District not found'}), 404
        
        lat, lon = DISTRICT_COORDS[district]
        
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum",
            "timezone": "Asia/Kolkata",
            "past_days": 92
        }
        
        response = req.get(url, params=params, timeout=10)
        data = response.json()
        
        precipitation = data['daily']['precipitation_sum']
        total_rainfall = sum(p for p in precipitation if p is not None)
        
        return jsonify({
            'district': district,
            'current_rainfall_mm': round(total_rainfall, 1),
            'latitude': lat,
            'longitude': lon,
            'period': 'Last 92 days'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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