"""
Flask application factory for the House Price ML Showcase Website.
Registers all blueprints, configures the app, and initializes the ML pipeline.
"""
import os
from flask import Flask


def create_app():
    """Create and configure the Flask application."""
    # Resolve paths relative to this file
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.abspath(os.path.join(base_dir, '..', 'data'))
    static_dir = os.path.join(base_dir, 'static')
    template_dir = os.path.join(base_dir, 'templates')

    app = Flask(
        __name__,
        static_folder=static_dir,
        template_folder=template_dir
    )

    app.config['SECRET_KEY'] = 'house-price-ml-showcase-2024'
    app.config['DATA_DIR'] = data_dir
    app.config['CHARTS_DIR'] = os.path.join(static_dir, 'images', 'charts')

    # Ensure the charts directory exists
    os.makedirs(app.config['CHARTS_DIR'], exist_ok=True)

    # Register blueprints
    from blueprints.main import main_bp
    from blueprints.eda import eda_bp
    from blueprints.cleaning import cleaning_bp
    from blueprints.features import features_bp
    from blueprints.models import models_bp
    from blueprints.explanation import explanation_bp
    from blueprints.prediction import prediction_bp
    from blueprints.results import results_bp
    from blueprints.docs import docs_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(eda_bp)
    app.register_blueprint(cleaning_bp)
    app.register_blueprint(features_bp)
    app.register_blueprint(models_bp)
    app.register_blueprint(explanation_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(results_bp)
    app.register_blueprint(docs_bp)

    # Initialize the ML pipeline on first request
    with app.app_context():
        _init_ml_pipeline(app)

    return app


def _init_ml_pipeline(app):
    """
    Train models and generate charts at startup, or load them from cache.
    Results are stored in app.config for use by blueprints.
    """
    import joblib
    
    base_dir = os.path.abspath(os.path.dirname(__file__))
    cache_path = os.path.join(base_dir, 'ml', 'saved_pipeline.joblib')
    
    if os.path.exists(cache_path):
        print("Loading pre-trained pipeline and models from cache...")
        result = joblib.load(cache_path)
    else:
        print("No cache found. Running ML pipeline and training models...")
        from ml.pipeline import build_pipeline
        result = build_pipeline(
            data_dir=app.config['DATA_DIR'],
            charts_dir=app.config['CHARTS_DIR']
        )
        # Save cache for future runs/production hosting
        try:
            joblib.dump(result, cache_path)
            print(f"Saved pipeline cache to {cache_path}")
        except Exception as e:
            print(f"Warning: Could not save pipeline cache: {e}")

    app.config['ML_PIPELINE'] = result['pipeline']
    app.config['ML_MODELS'] = result['models']
    app.config['ML_RESULTS'] = result['results']
    app.config['ML_FEATURE_NAMES'] = result['feature_names']
    app.config['ML_NUM_COLS'] = result['num_cols']
    app.config['ML_CAT_COLS'] = result['cat_cols']
    app.config['DATASET_STATS'] = result['dataset_stats']
