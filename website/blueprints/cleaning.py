"""
Data Cleaning page blueprint — serves the walkthrough of data cleaning steps.
"""
from flask import Blueprint, render_template

cleaning_bp = Blueprint('cleaning', __name__)


@cleaning_bp.route('/cleaning')
def cleaning():
    """Render the data cleaning walkthrough page."""
    return render_template('cleaning.html')
