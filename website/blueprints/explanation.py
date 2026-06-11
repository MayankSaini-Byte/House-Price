"""
Model Explanation page blueprint — beginner-friendly ML learning page.
"""
from flask import Blueprint, render_template

explanation_bp = Blueprint('explanation', __name__)


@explanation_bp.route('/explanation')
def explanation():
    """Render the ML explanation page."""
    return render_template('explanation.html')
