"""
Command Line Interface for FloodML
Provides CLI commands for training models and making flood predictions.
"""

import argparse
import sys
from typing import Optional
import structlog

logger = structlog.get_logger()


def predict() -> int:
    """
    CLI command for making flood predictions.
    
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    parser = argparse.ArgumentParser(
        description="Make flood predictions using trained FloodML models"
    )
    parser.add_argument(
        "--site", 
        type=str, 
        required=True,
        help="USGS site ID (e.g., 01438500)"
    )
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True,
        help="Path to trained model file"
    )
    parser.add_argument(
        "--days", 
        type=int, 
        default=7,
        help="Number of days to forecast (default: 7)"
    )
    parser.add_argument(
        "--output", 
        type=str,
        help="Output file path for predictions (optional)"
    )
    
    args = parser.parse_args()
    
    try:
        from floodml.prediction.forecaster import FloodForecaster
        
        logger.info("Loading model and making predictions", site=args.site)
        forecaster = FloodForecaster()
        forecaster.load_model(args.model_path)
        
        predictions = forecaster.predict(
            site_id=args.site,
            forecast_days=args.days
        )
        
        if args.output:
            predictions.to_csv(args.output)
            logger.info("Predictions saved", output_file=args.output)
        else:
            print(predictions)
            
        return 0
        
    except Exception as e:
        logger.error("Prediction failed", error=str(e))
        return 1


def train() -> int:
    """
    CLI command for training FloodML models.
    
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    parser = argparse.ArgumentParser(
        description="Train FloodML models for flood prediction"
    )
    parser.add_argument(
        "--site", 
        type=str, 
        required=True,
        help="USGS site ID (e.g., 01438500)"
    )
    parser.add_argument(
        "--model-type", 
        type=str, 
        choices=["random_forest", "lstm", "ensemble"],
        default="ensemble",
        help="Type of model to train (default: ensemble)"
    )
    parser.add_argument(
        "--start-date", 
        type=str, 
        required=True,
        help="Start date for training data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", 
        type=str, 
        required=True,
        help="End date for training data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--flood-stage", 
        type=float, 
        required=True,
        help="Flood stage threshold in feet"
    )
    parser.add_argument(
        "--output", 
        type=str,
        help="Output path for trained model (optional)"
    )
    
    args = parser.parse_args()
    
    try:
        # Import the appropriate model based on model_type
        if args.model_type == "random_forest":
            from floodml.models.random_forest import FloodRandomForest as Model
        elif args.model_type == "lstm":
            from floodml.models.lstm import FloodLSTM as Model
        else:  # ensemble
            from floodml.models.ensemble import FloodEnsemble as Model
        
        logger.info("Starting model training", 
                   site=args.site, 
                   model_type=args.model_type)
        
        model = Model()
        model.fit(
            site_id=args.site,
            start_date=args.start_date,
            end_date=args.end_date,
            flood_stage=args.flood_stage
        )
        
        if args.output:
            model.save(args.output)
            logger.info("Model saved", output_file=args.output)
        else:
            # Save to default location
            default_path = f"floodml_model_{args.site}_{args.model_type}.pkl"
            model.save(default_path)
            logger.info("Model saved", output_file=default_path)
            
        return 0
        
    except Exception as e:
        logger.error("Training failed", error=str(e))
        return 1


def main() -> Optional[int]:
    """
    Main entry point for CLI - this shouldn't be called directly.
    Use floodml-predict or floodml-train commands instead.
    """
    print("Use 'floodml-predict' or 'floodml-train' commands directly.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
