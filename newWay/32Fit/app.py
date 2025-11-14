from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
from typing import List, Dict

app = Flask(__name__)
CORS(app)

# Load model and encoders
model_path = 'model/bulkPredictor.pkl'
encoders = {
    'gender': joblib.load('model/le_gender.pkl'),
    'exercise': joblib.load('model/le_exercise.pkl'),
    'muscle_group': joblib.load('model/le_muscle_group.pkl'),
    'category': joblib.load('model/le_category.pkl'),
    'experience': joblib.load('model/le_experience.pkl')
}
model = joblib.load(model_path)

def get_baseline_params(age, gender, experience, current_size_cm, workout_time_years):
    if gender == 'M':
        base_limit = 45
        age_factor = max(0.7, 1 - (age - 25) * 0.01)
    else:
        base_limit = 30
        age_factor = max(0.7, 1 - (age - 25) * 0.008)
    M_max = base_limit * age_factor
    remaining_potential = max(0, M_max - (current_size_cm * 0.2))
    k_base = {'Beginner': 0.20, 'Intermediate': 0.10, 'Advanced': 0.05}
    k = k_base.get(experience, 0.10) * (1 / (1 + workout_time_years * 0.1))
    return remaining_potential, k

def calculate_baseline_growth(age, gender, experience, time_months, current_size_cm, workout_time_years):
    M_max, k = get_baseline_params(age, gender, experience, current_size_cm, workout_time_years)
    M0 = current_size_cm * 0.2
    baseline_growth = (M_max - M0) * (1 - np.exp(-k * time_months))
    return max(baseline_growth * 5, 0.1)

def safe_transform(encoder, value, name="value"):
    try:
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        print(f"{name} '{value}' not in training data")
        return 0
    except:
        return 0

@app.route('/')
def home():
    exercises = sorted(encoders['exercise'].classes_)
    html = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Bulk Predictor</title>
        <style>
            :root {{ --primary: #2563eb; --success: #10b981; --danger: #ef4444; }}
            body {{ font-family: 'Segoe UI', sans-serif; background: #f8fafc; margin: 0; padding: 20px; }}
            .container {{ max-width: 1000px; margin: auto; background: white; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); overflow: hidden; }}
            header {{ background: var(--primary); color: white; padding: 1.5rem; text-align: center; }}
            .form-section {{ padding: 2rem; }}
            .input-group {{ margin-bottom: 1rem; }}
            label {{ display: block; margin-bottom: 0.5rem; font-weight: 600; color: #374151; }}
            input, select {{ width: 100%; padding: 0.75rem; border: 1px solid #d1d5db; border-radius: 6px; font-size: 1rem; }}
            .exercise-item {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; background: #f9fafb; }}
            .btn {{ background: var(--primary); color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 6px; cursor: pointer; font-size: 1rem; }}
            .btn:hover {{ background: #1d4ed8; }}
            .add-btn {{ background: var(--success); margin-bottom: 1rem; }}
            .remove-btn {{ background: var(--danger); float: right; padding: 0.25rem 0.5rem; font-size: 0.8rem; }}
            .results {{ margin-top: 2rem; padding: 1.5rem; background: #f0f9ff; border-radius: 8px; }}
            .timeline {{ display: flex; gap: 1rem; margin-top: 1rem; }}
            .card {{ flex: 1; background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .loading {{ text-align: center; padding: 2rem; color: #6b7280; }}
            .error {{ color: var(--danger); margin-top: 1rem; }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Bulk Muscle Growth Predictor</h1>
                <p>Predict muscle gains over 3, 6, 9, and 12 months</p>
            </header>
            <div class="form-section">
                <form id="predictForm">
                    <div class="input-group">
                        <label>Age</label>
                        <input type="number" id="age" min="18" max="100" value="25" required>
                    </div>
                    <div class="input-group">
                        <label>Gender</label>
                        <select id="gender" required>
                            <option value="M">Male</option>
                            <option value="F">Female</option>
                        </select>
                    </div>
                    <div class="input-group">
                        <label>Experience Level</label>
                        <select id="experience" required>
                            <option>Beginner</option>
                            <option>Intermediate</option>
                            <option selected>Advanced</option>
                        </select>
                    </div>
                    <div class="input-group">
                        <label>Weekly Training Frequency (days)</label>
                        <input type="number" id="frequency" min="1" max="7" value="4" required>
                    </div>
                    <div class="input-group">
                        <label>Daily Protein (g)</label>
                        <input type="number" id="protein" min="50" max="300" value="150" required>
                    </div>
                    <div class="input-group">
                        <label>Daily Calories</label>
                        <input type="number" id="calories" min="1500" max="5000" value="2800" required>
                    </div>
                    <div class="input-group">
                        <label>Average Sleep (hours)</label>
                        <input type="number" step="0.1" id="sleep" min="5" max="12" value="7.5" required>
                    </div>
                    <div class="input-group">
                        <label>Current Muscle Size (cm², optional)</label>
                        <input type="number" step="0.1" id="current_size" value="0">
                    </div>
                    <div class="input-group">
                        <label>Years Training (optional)</label>
                        <input type="number" step="0.1" id="workout_years" value="0">
                    </div>

                    <h3>Exercises</h3>
                    <button type="button" class="btn add-btn" onclick="addExercise()">+ Add Exercise</button>
                    <div id="exercises-container">
                        <div class="exercise-item">
                            <select class="exercise-select" required>
                                <option value="">Select Exercise</option>
                                {''.join([f'<option>{ex}</option>' for ex in exercises])}
                            </select>
                            <input type="number" placeholder="Sets" class="sets" min="1" max="10" required>
                            <input type="number" placeholder="Reps" class="reps" min="1" max="20" required>
                            <input type="number" placeholder="Weight (kg)" class="weight" min="1" max="300" required>
                            <button type="button" class="btn remove-btn" onclick="this.parentElement.remove()">×</button>
                        </div>
                    </div>

                    <button type="submit" class="btn" style="width:100%; margin-top:1.5rem; font-size:1.1rem;">
                        Predict Growth
                    </button>
                </form>

                <div id="loading" class="loading" style="display:none;">Predicting...</div>
                <div id="error" class="error" style="display:none;"></div>
                <div id="results" class="results" style="display:none;"></div>
            </div>
        </div>

        <script>
            function addExercise() {{
                const container = document.getElementById('exercises-container');
                const div = document.createElement('div');
                div.className = 'exercise-item';
                div.innerHTML = `
                    <select class="exercise-select" required>
                        <option value="">Select Exercise</option>
                        {''.join([f'<option>{ex}</option>' for ex in exercises])}
                    </select>
                    <input type="number" placeholder="Sets" class="sets" min="1" max="10" required>
                    <input type="number" placeholder="Reps" class="reps" min="1" max="20" required>
                    <input type="number" placeholder="Weight (kg)" class="weight" min="1" max="300" required>
                    <button type="button" class="btn remove-btn" onclick="this.parentElement.remove()">×</button>
                `;
                container.appendChild(div);
            }}

            document.getElementById('predictForm').onsubmit = async (e) => {{
                e.preventDefault();
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                document.getElementById('error').style.display = 'none';

                const exercises = Array.from(document.querySelectorAll('.exercise-item')).map(item => ({{
                    exercise_name: item.querySelector('.exercise-select').value,
                    sets: parseInt(item.querySelector('.sets').value),
                    reps: parseInt(item.querySelector('.reps').value),
                    weight: parseInt(item.querySelector('.weight').value),
                    target_muscle_group: '',
                    exercise_category: 'Compound'
                }}));

                if (exercises.some(ex => !ex.exercise_name)) {{
                    showError('Please select an exercise for each entry.');
                    return;
                }}

                const data = {{
                    age: parseInt(document.getElementById('age').value),
                    gender: document.getElementById('gender').value,
                    experience: document.getElementById('experience').value,
                    frequency: parseInt(document.getElementById('frequency').value),
                    protein: parseFloat(document.getElementById('protein').value),
                    calories: parseInt(document.getElementById('calories').value),
                    sleep: parseFloat(document.getElementById('sleep').value),
                    current_size_cm: parseFloat(document.getElementById('current_size').value) || 0,
                    workout_time_years: parseFloat(document.getElementById('workout_years').value) || 0,
                    exercises: exercises
                }};

                try {{
                    const res = await fetch('/bulk', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify(data)
                    }});
                    const result = await res.json();
                    if (res.ok) {{
                        displayResults(result.result);
                    }} else {{
                        showError(result.error || 'Prediction failed');
                    }}
                }} catch (err) {{
                    showError('Network error: ' + err.message);
                }}
                document.getElementById('loading').style.display = 'none';
            }};

            function showError(msg) {{
                document.getElementById('loading').style.display = 'none';
                const el = document.getElementById('error');
                el.textContent = msg;
                el.style.display = 'block';
            }}

            function displayResults(data) {{
                const container = document.getElementById('results');
                container.innerHTML = '<h3>Predicted Muscle Growth (cm²)</h3><div class="timeline">' +
                    Object.entries(data).map(([months, preds]) => `
                        <div class="card">
                            <h4>${{months}} Months</h4>
                            ${{Object.entries(preds).map(([k, v]) => `<div><strong>${{k}}</strong>: ${{v}}</div>`).join('')}}
                        </div>
                    `).join('') + '</div>';
                container.style.display = 'block';
            }}
        </script>
    </body>
    </html>
    '''
    return render_template_string(html)

@app.route('/bulk', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        intervals = [3, 6, 9, 12]
        result = {}

        for interval in intervals:
            predictions = predict_growth_for_interval(data, interval)
            result[str(interval)] = predictions

        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def predict_growth_for_interval(data, time_months):
    age = data['age']
    gender = data['gender']
    experience = data['experience']
    current_size_cm = data.get('current_size_cm', 0)
    workout_time_years = data.get('workout_time_years', 0)

    muscle_groups = {}
    for ex in data['exercises']:
        name = ex['exercise_name']
        # Map exercise to muscle group (simplified)
        mg_map = {
            'Chest': ['Bench', 'Press', 'Flyes', 'Dips', 'Push-ups'],
            'Back': ['Rows', 'Pulldowns', 'Pull-ups', 'Deadlift'],
            'Arms': ['Curls', '21s']
        }
        muscle = 'Unknown'
        for m, keywords in mg_map.items():
            if any(k in name for k in keywords):
                muscle = m
                break
        if muscle not in muscle_groups:
            muscle_groups[muscle] = []
        muscle_groups[muscle].append(ex)

    predictions = {}
    for muscle, ex_list in muscle_groups.items():
        total_volume = sum(ex['sets'] * ex['reps'] * ex['weight'] for ex in ex_list)
        avg_sets = np.mean([ex['sets'] for ex in ex_list])
        avg_reps = np.mean([ex['reps'] for ex in ex_list])
        equiv_weight = total_volume / (avg_sets * avg_reps) if avg_sets and avg_reps else 50

        baseline = calculate_baseline_growth(age, gender, experience, time_months, current_size_cm, workout_time_years)

        feature_row = {
            'age': age,
            'gender_encoded': safe_transform(encoders['gender'], gender),
            'exercise_name_encoded': safe_transform(encoders['exercise'], ex_list[0]['exercise_name']),
            'sets': avg_sets,
            'reps': avg_reps,
            'weight': equiv_weight,
            'frequency': data['frequency'],
            'protein': data['protein'],
            'calories': data['calories'],
            'sleep': data['sleep'],
            'experience_encoded': safe_transform(encoders['experience'], experience),
            'muscle_group_encoded': 0,
            'category_encoded': 0,
            'protein_per_kg': data['protein'] / 70,
            'volume': avg_sets * avg_reps,
            'intensity': equiv_weight / avg_reps,
            'calories_per_kg': data['calories'] / 70,
            'genetic_advantage': 3
        }

        X = pd.DataFrame([feature_row])
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)
        adj_factor = model.predict(X)[0]
        growth = baseline * adj_factor
        predictions[f"{muscle} growth"] = f"{growth:.2f} cm2"

    return predictions

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=False)
