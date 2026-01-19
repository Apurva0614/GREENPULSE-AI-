from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load models
regressor = None
classifier = None
models_loaded = False

try:
    # Try multiple paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = os.getcwd()
    
    # Try paths in order
    possible_paths = [
        os.path.join(base_dir, "random_forest_regressor.pkl"),
        os.path.join(current_dir, "random_forest_regressor.pkl"),
        "random_forest_regressor.pkl"
    ]
    
    regressor_path = None
    classifier_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            regressor_path = path
            classifier_path = path.replace("regressor", "classifier")
            if os.path.exists(classifier_path):
                break
    
    if regressor_path and classifier_path and os.path.exists(regressor_path) and os.path.exists(classifier_path):
        try:
            regressor = joblib.load(regressor_path)
            classifier = joblib.load(classifier_path)
            models_loaded = True
            print("✓ Models loaded successfully!")
            print(f"  Regressor: {regressor_path}")
            print(f"  Classifier: {classifier_path}")
        except ValueError as e:
            if "incompatible dtype" in str(e) or "node array" in str(e):
                print("ERROR: Model version incompatibility detected!")
                print("The models were saved with a different version of scikit-learn.")
                print("Please retrain the models using the current scikit-learn version.")
                print(f"Error details: {str(e)}")
                models_loaded = False
            else:
                raise
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise
    else:
        print(f"Warning: Model files not found.")
        print(f"Base directory: {base_dir}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in base directory: {[f for f in os.listdir(base_dir) if f.endswith('.pkl')]}")
        print(f"Files in current directory: {[f for f in os.listdir('.') if f.endswith('.pkl')]}")
except Exception as e:
    models_loaded = False
    print(f"Error loading models: {str(e)}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()

# Load data
data = None
data_loaded = False

try:
    # Try multiple paths for data file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = os.getcwd()
    
    possible_paths = [
        os.path.join(base_dir, "feature_engineered_greenpulse.csv"),
        os.path.join(current_dir, "feature_engineered_greenpulse.csv"),
        "feature_engineered_greenpulse.csv"
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path:
        data = pd.read_csv(data_path)
        data_loaded = True
        print("✓ Data loaded successfully!")
        print(f"  Data file: {data_path}")
    else:
        print(f"Warning: Data file not found.")
        print(f"Tried paths: {possible_paths}")
except Exception as e:
    data_loaded = False
    print(f"Error loading data: {str(e)}")
    import traceback
    traceback.print_exc()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data/summary')
def data_summary():
    if not data_loaded:
        return jsonify({"error": "Data not loaded"}), 500
    
    summary = {
        "total_rows": len(data),
        "total_columns": len(data.columns),
        "countries": int(data['country'].nunique()) if 'country' in data.columns else 0,
        "gdi_stats": {
            "mean": float(data['GDI'].mean()) if 'GDI' in data.columns else 0,
            "min": float(data['GDI'].min()) if 'GDI' in data.columns else 0,
            "max": float(data['GDI'].max()) if 'GDI' in data.columns else 0
        },
        "category_distribution": data['GDI_Category'].value_counts().to_dict() if 'GDI_Category' in data.columns else {}
    }
    return jsonify(summary)

@app.route('/api/data/timeline')
def timeline_data():
    if not data_loaded:
        return jsonify({"error": "Data not loaded"}), 500
    
    loss_cols = [col for col in data.columns if col.startswith('tc_loss_ha_')]
    years = [int(col.split('_')[-1]) for col in loss_cols]
    total_loss = data[loss_cols].sum().tolist()
    
    return jsonify({
        "years": years,
        "total_loss": total_loss
    })

@app.route('/api/data/categories')
def category_data():
    if not data_loaded:
        return jsonify({"error": "Data not loaded"}), 500
    
    categories = data['GDI_Category'].value_counts().to_dict()
    return jsonify(categories)

@app.route('/api/model/status')
def model_status():
    """Endpoint to check model loading status"""
    status = {
        "models_loaded": models_loaded,
        "regressor_loaded": regressor is not None,
        "classifier_loaded": classifier is not None,
        "data_loaded": data_loaded,
        "error": None
    }
    
    if not models_loaded:
        status["error"] = "Models failed to load. This is likely due to a scikit-learn version mismatch. The models were saved with scikit-learn 1.0.2 but you're using a different version. Please retrain the models using: python ml.ipynb or your training script."
    
    return jsonify(status)

@app.route('/api/predict', methods=['POST'])
def predict():
    if not models_loaded or regressor is None or classifier is None:
        return jsonify({
            "error": "Models not loaded. This is likely due to a scikit-learn version mismatch. The models need to be retrained with your current scikit-learn version. Check /api/model/status for more details.",
            "success": False
        }), 500
    
    if not data_loaded or data is None:
        return jsonify({
            "error": "Data not loaded. Cannot determine feature names.",
            "success": False
        }), 500
    
    try:
        input_data = request.json
        
        # Create feature vector
        features = {}
        
        # Get all numeric columns from training data (excluding GDI)
        numeric_cols = data.select_dtypes(include=np.number).columns
        feature_cols = [col for col in numeric_cols if col != 'GDI']
        
        # Calculate derived features if we have the base inputs
        # Normalize key names for matching
        normalized_input = {}
        for key, value in input_data.items():
            # Try both with dash and underscore
            if key in feature_cols:
                normalized_input[key] = value
            elif key.replace('-', '_') in feature_cols:
                normalized_input[key.replace('-', '_')] = value
            elif key.replace('_', '-') in feature_cols:
                normalized_input[key.replace('_', '-')] = value
            else:
                normalized_input[key] = value
        
        # Calculate extent_change if we have both extents
        if 'extent_2000_ha' in normalized_input and 'extent_2010_ha' in normalized_input:
            normalized_input['extent_change_2000_2010'] = float(normalized_input['extent_2010_ha']) - float(normalized_input['extent_2000_ha'])
        
        # Calculate total tree loss if we have individual year losses
        provided_losses = {k: v for k, v in normalized_input.items() if k.startswith('tc_loss_ha_')}
        if provided_losses:
            # If only one year is provided, estimate total loss proportionally
            if 'tc_loss_ha_2023' in normalized_input:
                single_year_loss = float(normalized_input.get('tc_loss_ha_2023', 0))
                # Use average ratio from training data
                avg_ratio = data['total_tree_loss_ha_2001_2023'].sum() / data['tc_loss_ha_2023'].sum() if data['tc_loss_ha_2023'].sum() > 0 else 20
                estimated_total = single_year_loss * avg_ratio
                normalized_input['total_tree_loss_ha_2001_2023'] = estimated_total
        
        # Calculate gain_loss_ratio if we have both
        gain_key = 'gain_2000-2020_ha' if 'gain_2000-2020_ha' in normalized_input else 'gain_2000_2020_ha'
        if gain_key in normalized_input and 'total_tree_loss_ha_2001_2023' in normalized_input:
            gain = float(normalized_input.get(gain_key, 0))
            total_loss = float(normalized_input.get('total_tree_loss_ha_2001_2023', 1))
            normalized_input['gain_loss_ratio'] = gain / (total_loss + 1)  # +1 to avoid division by zero
        
        input_data = normalized_input
        
        # Set default values to median from training data (better than zeros)
        for col in feature_cols:
            if col in data.columns:
                # Use median instead of 0 for better predictions
                features[col] = float(data[col].median())
            else:
                features[col] = 0
        
        # Update with provided values
        for key, value in input_data.items():
            # Try exact match first
            if key in features:
                features[key] = float(value)
            else:
                # Try with underscore (for gain_2000-2020_ha -> gain_2000_2020_ha)
                normalized_key = key.replace('-', '_')
                if normalized_key in features:
                    features[normalized_key] = float(value)
                # Try with dash (for gain_2000_2020_ha -> gain_2000-2020_ha)
                elif key.replace('_', '-') in features:
                    features[key.replace('_', '-')] = float(value)
        
        # Create DataFrame
        input_df = pd.DataFrame([features])
        input_df = input_df.reindex(columns=feature_cols, fill_value=0)
        
        # Make GDI prediction using regression model
        gdi_prediction = float(regressor.predict(input_df)[0])
        
        # Categorize based on predicted GDI (more reliable than using classifier with incomplete features)
        def categorize_gdi(gdi_value):
            if gdi_value <= -5:
                return "Excellent (Net Gain)"
            elif -5 < gdi_value <= 0:
                return "Acceptable"
            elif 0 < gdi_value <= 10:
                return "Concerning"
            else:
                return "High-Risk"
        
        category_prediction = categorize_gdi(gdi_prediction)
        
        return jsonify({
            "gdi": round(gdi_prediction, 3),
            "category": str(category_prediction),
            "success": True
        })
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 400

@app.route('/api/model/info')
def model_info():
    if not models_loaded or data is None:
        return jsonify({"error": "Models or data not loaded"}), 500
    
    return jsonify({
        "regression_model": "Random Forest Regressor",
        "classification_model": "Random Forest Classifier",
        "regression_r2": 0.984,
        "regression_mae": 3.148,
        "classification_accuracy": 0.981,
        "features_count": len(data.select_dtypes(include=np.number).columns) - 1
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)

