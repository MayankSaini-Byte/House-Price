"""
Entry point for the House Price Prediction ML Showcase Website.
Run: python run.py
"""
from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
