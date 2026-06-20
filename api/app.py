# CropLens - Flask Backend API
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
from flask import send_file
import io
import os
import joblib
import numpy as np
import pandas as pd
import requests as req
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from huggingface_hub import hf_hub_download

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


def ensure_models():
    model_path = os.path.join(MODELS, 'croplens_model.pkl')
    if not os.path.exists(model_path):
        print("Downloading models from Hugging Face...")
        os.makedirs(MODELS, exist_ok=True)
        os.makedirs(PROCESSED, exist_ok=True)

        model_files = [
            'models/croplens_model.pkl',
            'models/label_encoder.pkl',
            'models/crop_encoder.pkl',
            'models/soil_encoder.pkl',
            'models/season_encoder.pkl',
            'models/district_encoder.pkl',
            'models/feature_columns.pkl',
            'models/shap_explainer.pkl',
        ]

        data_files = [
            'data/processed/master_dataset.csv',
            'data/processed/maharashtra_crops_clean.csv',
            'data/processed/rainfall_clean.csv',
        ]

        for file_path in model_files + data_files:
            local_path = os.path.join(BASE, file_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            hf_hub_download(
                repo_id='rihan077/CropLens',
                repo_type='space',
                filename=file_path,
                local_dir=BASE
            )
            print(f'Downloaded: {file_path}')

        print("✅ All files downloaded")


ensure_models()

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

# Feature display names for PDF
FEATURE_DISPLAY_NAMES = {
    'annual_rainfall': 'Annual Rainfall (mm)',
    'monsoon_rainfall': 'Monsoon Rainfall (mm)',
    'rainfall_deviation_pct': 'Rainfall Deviation %',
    'rain_june': 'June Rainfall (mm)',
    'rain_july': 'July Rainfall (mm)',
    'rain_august': 'August Rainfall (mm)',
    'rain_september': 'September Rainfall (mm)',
    'normal_annual_rainfall': 'Normal Annual Rainfall',
    'drought_streak': 'Drought Streak (years)',
    'yield_lag_1': 'Last Year Yield (kg/ha)',
    'yield_lag_2': '2 Years Ago Yield (kg/ha)',
    'yield_lag_3': '3 Years Ago Yield (kg/ha)',
    'yield_3yr_avg': '3 Year Avg Yield (kg/ha)',
    'yield_trend': 'Yield Trend',
    'yield_volatility': 'Yield Volatility',
    'compatibility_score': 'Soil Compatibility Score',
    'msp_per_quintal': 'MSP Price (Rs/quintal)',
    'msp_growth_rate': 'MSP Growth Rate %',
    'area_hectares': 'Cultivated Area (hectares)',
    'crop_encoded': 'Crop Type',
    'soil_encoded': 'Soil Type',
    'season_encoded': 'Season',
    'district_encoded': 'District'
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
    latest = master_df.sort_values('year').groupby(
        'district').last().reset_index()

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


@app.route('/api/generate-pdf', methods=['POST'])
def generate_pdf():
    try:
        data = request.get_json()
        district = data.get('district', '')
        crop = data.get('crop', '')
        season = data.get('season', '')

        result, error = get_risk_details(district, crop, season)
        if error:
            return jsonify({'error': error}), 404

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                rightMargin=50, leftMargin=50,
                                topMargin=50, bottomMargin=50)

        green = HexColor('#1B5E20')
        red = HexColor('#C62828')
        orange = HexColor('#E65100')

        if result['risk_level'] == 'High Risk':
            risk_color = red
        elif result['risk_level'] == 'Medium Risk':
            risk_color = orange
        else:
            risk_color = green

        elements = []

        title_style = ParagraphStyle(
            'Title', fontSize=20, textColor=green,
            spaceAfter=6, fontName='Helvetica-Bold'
        )
        elements.append(Paragraph(
            'CropLens Risk Assessment Report', title_style))
        elements.append(Paragraph(
            'Agricultural Risk Intelligence System for Maharashtra',
            ParagraphStyle('Sub', fontSize=10,
                           textColor=HexColor('#666666'), spaceAfter=20)
        ))
        elements.append(Spacer(1, 0.2 * inch))

        input_data = [
            ['Field', 'Value'],
            ['District', result['district']],
            ['Crop', result['crop']],
            ['Season', result['season']],
            ['Year', result['year']],
            ['Soil Type', result['soil_type']],
        ]

        input_table = Table(input_data, colWidths=[2 * inch, 4 * inch])
        input_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), green),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [HexColor('#F5F7F5'), HexColor('#FFFFFF')]),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#E0E0E0')),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(input_table)
        elements.append(Spacer(1, 0.3 * inch))

        risk_style = ParagraphStyle(
            'Risk', fontSize=16, textColor=risk_color,
            spaceAfter=6, fontName='Helvetica-Bold'
        )
        elements.append(Paragraph(
            f'Risk Level: {result["risk_level"]}', risk_style))
        elements.append(Paragraph(
            f'Confidence: {result["confidence"]}%',
            ParagraphStyle('Conf', fontSize=12, spaceAfter=20)
        ))
        elements.append(Spacer(1, 0.2 * inch))

        stats_data = [
            ['Metric', 'Value'],
            ['Predicted Yield', f'{result["predicted_yield"]} kg/ha'],
            ['District Avg Yield', f'{result["district_avg_yield"]} kg/ha'],
            ['Annual Rainfall', f'{result["annual_rainfall"]} mm'],
            ['Rainfall Deviation', f'{result["rainfall_deviation"]}%'],
            ['Drought Streak', f'{result["drought_streak"]} years'],
            ['Soil Compatibility', str(result["compatibility_score"])],
        ]

        stats_table = Table(stats_data, colWidths=[3 * inch, 3 * inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), green),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [HexColor('#F5F7F5'), HexColor('#FFFFFF')]),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#E0E0E0')),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(stats_table)
        elements.append(Spacer(1, 0.3 * inch))

        elements.append(Paragraph(
            'Key Risk Factors (SHAP Analysis)',
            ParagraphStyle('H2', fontSize=13, textColor=green,
                           fontName='Helvetica-Bold', spaceAfter=10)
        ))

        reasons_data = [['Factor', 'Value', 'Impact']]
        for r in result['reasons']:
            direction = 'Increases Risk' if r['direction'] == 'increases_risk' \
                else 'Reduces Risk'
            feature_name = FEATURE_DISPLAY_NAMES.get(
                r['feature'], r['feature'].replace('_', ' ').title()
            )
            display_value = str(r['value'])
            if r['feature'] == 'crop_encoded':
                display_value = result['crop']
            elif r['feature'] == 'soil_encoded':
                display_value = result['soil_type']
            elif r['feature'] == 'season_encoded':
                display_value = result['season']
            elif r['feature'] == 'district_encoded':
                display_value = result['district']

            reasons_data.append([
                feature_name,
                display_value,
                direction
            ])

        reasons_table = Table(reasons_data,
                              colWidths=[2.5 * inch, 1.5 * inch, 2 * inch])
        reasons_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), green),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [HexColor('#F5F7F5'), HexColor('#FFFFFF')]),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#E0E0E0')),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(reasons_table)
        elements.append(Spacer(1, 0.3 * inch))

        if result['risk_level'] == 'High Risk':
            rec = (f"High caution advised for {crop} in {district}. "
                   f"Consider crop insurance before planting. "
                   f"Rainfall is {result['rainfall_deviation']}% from normal. "
                   f"Recommend drought-resistant varieties.")
        elif result['risk_level'] == 'Medium Risk':
            rec = (f"Moderate risk for {crop} in {district}. "
                   f"Normal precautions advised. "
                   f"Monitor rainfall patterns closely during growing season.")
        else:
            rec = (f"Favorable conditions for {crop} in {district}. "
                   f"Good season expected. "
                   f"Standard farming practices should yield good results.")

        elements.append(Paragraph(
            'Recommendation',
            ParagraphStyle('H2', fontSize=13, textColor=green,
                           fontName='Helvetica-Bold', spaceAfter=6)
        ))
        elements.append(Paragraph(
            rec,
            ParagraphStyle('Body', fontSize=10, spaceAfter=20, leading=16)
        ))

        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(
            'Generated by CropLens - Agricultural Risk Intelligence System | '
            'Built with LightGBM + SHAP | Data: Government of India',
            ParagraphStyle('Footer', fontSize=8,
                           textColor=HexColor('#999999'), alignment=1)
        ))

        doc.build(elements)
        buffer.seek(0)

        filename = f'CropLens_{district}_{crop}_{season}.pdf'.replace(
            ' ', '_')
        return send_file(
            buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )

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
    port = int(os.environ.get('PORT', 7860))
    app.run(debug=False, host='0.0.0.0', port=port)