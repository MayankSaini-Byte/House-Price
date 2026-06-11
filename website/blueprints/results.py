"""
Results & Insights page blueprint.
"""
from flask import Blueprint, render_template, current_app

results_bp = Blueprint('results', __name__)


@results_bp.route('/results')
def results():
    """Render the results and insights page."""
    ml_results = current_app.config.get('ML_RESULTS', {})
    stats = current_app.config.get('DATASET_STATS', {})
    return render_template('results.html', results=ml_results, stats=stats)
