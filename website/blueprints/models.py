"""
Model Comparison page blueprint.
"""
from flask import Blueprint, render_template, current_app

models_bp = Blueprint('models', __name__)


@models_bp.route('/models')
def models():
    """Render the model comparison page with results for each model."""
    results = current_app.config.get('ML_RESULTS', {})
    return render_template('models.html', results=results)
