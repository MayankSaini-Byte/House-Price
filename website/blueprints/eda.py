"""
EDA page blueprint — serves dataset overview and exploratory charts.
"""
from flask import Blueprint, render_template, current_app

eda_bp = Blueprint('eda', __name__)


@eda_bp.route('/eda')
def eda():
    """Render the EDA page with dataset statistics and chart paths."""
    stats = current_app.config.get('DATASET_STATS', {})
    return render_template('eda.html', stats=stats)
