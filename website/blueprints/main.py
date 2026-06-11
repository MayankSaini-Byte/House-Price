"""
Landing page blueprint — serves the hero section, stats, and architecture diagram.
"""
from flask import Blueprint, render_template, current_app

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def landing():
    """Render the landing page with project statistics."""
    stats = current_app.config.get('DATASET_STATS', {})
    results = current_app.config.get('ML_RESULTS', {})
    return render_template('landing.html', stats=stats, results=results)
