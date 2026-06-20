// ─────────────────────────────────────────
// CropLens - Main JavaScript
// ─────────────────────────────────────────

// Load dropdowns on page load
document.addEventListener('DOMContentLoaded', function() {
    loadDistricts();
    loadCrops();
    loadSeasons();
});

// Fetch live weather for selected district
async function fetchWeather(district) {
    try {
        const res = await fetch(`/api/weather/${district}`);
        const data = await res.json();
        
        if (data.current_rainfall_mm !== undefined) {
            let weatherDiv = document.getElementById('weather-info');
            if (!weatherDiv) {
                weatherDiv = document.createElement('div');
                weatherDiv.id = 'weather-info';
                weatherDiv.style.cssText = `
                    background: #E3F2FD;
                    border-left: 4px solid #1565C0;
                    border-radius: 8px;
                    padding: 0.8rem 1rem;
                    margin-top: 1rem;
                    font-size: 0.9rem;
                    color: #1A237E;
                `;
                document.querySelector('.form-card').appendChild(weatherDiv);
            }
            weatherDiv.innerHTML = `
                🌧️ <strong>Live Rainfall Data for ${district}:</strong> 
                ${data.current_rainfall_mm} mm (last 92 days)
            `;
        }
    } catch (err) {
        console.log('Weather fetch failed:', err);
    }
}

// Load districts
async function loadDistricts() {
    try {
        const res = await fetch('/api/districts');
        const data = await res.json();
        const select = document.getElementById('district');
        data.districts.forEach(d => {
            const option = document.createElement('option');
            option.value = d;
            option.textContent = d;
            select.appendChild(option);
        });
        // Fetch weather when district changes
        select.addEventListener('change', function() {
            if (this.value) fetchWeather(this.value);
        });
    } catch (err) {
        console.error('Error loading districts:', err);
    }
}

// Load crops
async function loadCrops() {
    try {
        const res = await fetch('/api/crops');
        const data = await res.json();
        const select = document.getElementById('crop');
        data.crops.forEach(c => {
            const option = document.createElement('option');
            option.value = c;
            option.textContent = c;
            select.appendChild(option);
        });
    } catch (err) {
        console.error('Error loading crops:', err);
    }
}

// Load seasons
async function loadSeasons() {
    try {
        const res = await fetch('/api/seasons');
        const data = await res.json();
        const select = document.getElementById('season');
        data.seasons.forEach(s => {
            const option = document.createElement('option');
            option.value = s;
            option.textContent = s;
            select.appendChild(option);
        });
    } catch (err) {
        console.error('Error loading seasons:', err);
    }
}

// Main predict function
async function predict() {
    const district = document.getElementById('district').value;
    const crop = document.getElementById('crop').value;
    const season = document.getElementById('season').value;

    // Validate
    if (!district || !crop || !season) {
        alert('Please select District, Crop and Season');
        return;
    }

    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('result-card').style.display = 'none';
    document.getElementById('error-card').style.display = 'none';

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ district, crop, season })
        });

        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.error || 'Prediction failed');
        }

        displayResult(data);

    } catch (err) {
        document.getElementById('error-card').style.display = 'block';
        document.getElementById('error-text').textContent = 
            '❌ Error: ' + err.message;
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

// Display result
function displayResult(data) {
    const card = document.getElementById('result-card');
    card.style.display = 'block';

    // Risk level setup
    let emoji, colorClass, bgClass, recommendation;

    if (data.risk_level === 'High Risk') {
        emoji = '🔴';
        colorClass = 'high-risk';
        bgClass = 'high-risk-bg';
        recommendation = `High caution advised for ${data.crop} in ${data.district}. 
        Consider crop insurance before planting. 
        Rainfall is ${data.rainfall_deviation}% from normal. 
        Recommend drought-resistant varieties or reducing sown area.`;
    } else if (data.risk_level === 'Medium Risk') {
        emoji = '🟡';
        colorClass = 'medium-risk';
        bgClass = 'medium-risk-bg';
        recommendation = `Moderate risk for ${data.crop} in ${data.district}. 
        Normal precautions advised. 
        Monitor rainfall patterns closely during growing season. 
        Standard crop insurance recommended.`;
    } else {
        emoji = '🟢';
        colorClass = 'low-risk';
        bgClass = 'low-risk-bg';
        recommendation = `Favorable conditions for ${data.crop} in ${data.district}. 
        Good season expected based on historical patterns. 
        Standard farming practices should yield good results.`;
    }

    // Set result header
    document.getElementById('result-header').className = 
        'result-header ' + bgClass;
    document.getElementById('risk-badge').textContent = emoji;
    document.getElementById('risk-title').textContent = data.risk_level;
    document.getElementById('risk-title').className = colorClass;
    document.getElementById('risk-subtitle').textContent = 
        `${data.district} | ${data.crop} | ${data.season} | ${data.year}`;

    // Set stats
    document.getElementById('confidence').textContent = 
        data.confidence + '%';
    document.getElementById('predicted-yield').textContent = 
        data.predicted_yield + ' kg/ha';
    document.getElementById('avg-yield').textContent = 
        data.district_avg_yield + ' kg/ha';
    document.getElementById('rainfall').textContent = 
        data.annual_rainfall + ' mm';
    document.getElementById('rain-deviation').textContent = 
        data.rainfall_deviation + '%';
    document.getElementById('soil-type').textContent = 
        data.soil_type;

    // Set reasons
    const reasonsList = document.getElementById('reasons-list');
    reasonsList.innerHTML = '';

    data.reasons.forEach(reason => {
        const isIncrease = reason.direction === 'increases_risk';
        const div = document.createElement('div');
        div.className = 'reason-item ' + 
            (isIncrease ? 'reason-increases' : 'reason-reduces');
        div.innerHTML = `
            <span class="reason-icon">${isIncrease ? '↑' : '↓'}</span>
            <span class="reason-text">
                <span class="reason-feature">${formatFeature(reason.feature)}</span>: 
                ${reason.value.toFixed(1)} 
                — ${isIncrease ? 'Increases Risk' : 'Reduces Risk'}
            </span>
        `;
        reasonsList.appendChild(div);
    });

    // Set recommendation
    document.getElementById('recommendation-text').textContent = 
        recommendation;

    // Scroll to result
    card.scrollIntoView({ behavior: 'smooth' });
}

// Format feature names for display
function formatFeature(feature) {
    const names = {
        'annual_rainfall': 'Annual Rainfall',
        'monsoon_rainfall': 'Monsoon Rainfall',
        'rainfall_deviation_pct': 'Rainfall Deviation',
        'rain_june': 'June Rainfall',
        'rain_july': 'July Rainfall',
        'rain_august': 'August Rainfall',
        'rain_september': 'September Rainfall',
        'normal_annual_rainfall': 'Normal Annual Rainfall',
        'drought_streak': 'Drought Streak Years',
        'yield_lag_1': 'Last Year Yield',
        'yield_lag_2': '2 Years Ago Yield',
        'yield_lag_3': '3 Years Ago Yield',
        'yield_3yr_avg': '3 Year Average Yield',
        'yield_trend': 'Yield Trend',
        'yield_volatility': 'Yield Volatility',
        'compatibility_score': 'Soil Compatibility Score',
        'msp_per_quintal': 'MSP Price',
        'msp_growth_rate': 'MSP Growth Rate',
        'area_hectares': 'Cultivated Area',
        'crop_encoded': 'Crop Type',
        'soil_encoded': 'Soil Type',
        'season_encoded': 'Season',
        'district_encoded': 'District'
    };
    return names[feature] || feature;
}