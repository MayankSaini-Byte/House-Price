"""
Interactive Prediction page blueprint — form + API endpoint.
"""
from flask import Blueprint, render_template, request, jsonify, current_app
from ml.predict import predict_price

prediction_bp = Blueprint('prediction', __name__)


@prediction_bp.route('/predict')
def predict_page():
    """Render the interactive prediction form page."""
    return render_template('prediction.html')


@prediction_bp.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for house price prediction.
    Accepts JSON with feature values, returns predicted price.
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400

    pipeline = current_app.config.get('ML_PIPELINE')
    models = current_app.config.get('ML_MODELS', {})
    num_cols = current_app.config.get('ML_NUM_COLS', [])
    cat_cols = current_app.config.get('ML_CAT_COLS', [])

    # Use the best performing model (Linear Regression based on notebook)
    best_model_name = current_app.config.get('DATASET_STATS', {}).get('best_model', 'Linear Regression')
    model = models.get(best_model_name)

    if not pipeline or not model:
        return jsonify({'success': False, 'error': 'Model not ready'}), 503

    # Build input dict: convert form strings to proper types
    input_data = {}
    for key, value in data.items():
        try:
            input_data[key] = float(value)
        except (ValueError, TypeError):
            input_data[key] = value

    result = predict_price(pipeline, model, input_data, num_cols, cat_cols)
    return jsonify(result)
