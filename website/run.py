"""
Entry point for the House Price Prediction ML Showcase Website.
Run: python run.py
"""
import os
import sys

# Ensure the website directory is in Python's search path (for Vercel deployment)
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
