"""
Simple example of using FloodML for flood prediction
"""

from floodml import FloodPredictor
from datetime import datetime, timedelta

def basic_flood_prediction_example():
    """
    Basic example showing how to use FloodML for flood prediction
    """
    print("🌊 FloodML Basic Example")
    print("=" * 50)
    
    # Initialize predictor for Delaware River at Montague, NJ
    predictor = FloodPredictor(
        usgs_site="01438500",
        flood_stage_ft=25.0,  # NOAA flood stage for this gauge
        model="rf"            # Random Forest (fastest for demo)
    )
    
    print(f"✅ Initialized predictor for USGS site {predictor.usgs_site}")
    print(f"📍 Location: {predictor.lat:.4f}, {predictor.lon:.4f}")
    print(f"⚠️  Flood stage: {predictor.flood_stage_ft} feet")
    
    # Train model on recent data (last 2 years)
    print("\n📚 Training model...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    try:
        predictor.fit(
            start_date=start_date,
            end_date=end_date,
            forecast_hours=24
        )
        print("✅ Model training completed")
        
        # Get model performance info
        model_info = predictor.get_model_performance()
        print(f"📊 Model type: {model_info.get('type', 'Unknown')}")
        print(f"📈 Features: {model_info.get('num_features', 0)}")
        
        # Make 24-hour flood prediction
        print("\n🔮 Making flood prediction...")
        prediction = predictor.predict_flood_probability(hours_ahead=24)
        
        print(f"🎯 Flood probability (24h): {prediction.probability:.1%}")
        
        if prediction.probability > 0.7:
            print("🚨 HIGH FLOOD RISK!")
        elif prediction.probability > 0.3:
            print("⚠️  Moderate flood risk")
        else:
            print("✅ Low flood risk")
        
        # Get current conditions
        print("\n📊 Current Conditions:")
        conditions = predictor.get_current_conditions()
        
        if conditions.get('current_stage_ft'):
            current_stage = conditions['current_stage_ft']
            flood_stage = conditions['flood_stage_ft']
            print(f"💧 Current stage: {current_stage:.2f} ft")
            print(f"📏 Above flood stage: {current_stage - flood_stage:.2f} ft")
        
        if conditions.get('current_discharge_cfs'):
            print(f"🌊 Current discharge: {conditions['current_discharge_cfs']:,.0f} cfs")
        
        print(f"⏰ Data timestamp: {conditions.get('data_timestamp', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 This might happen if USGS data is temporarily unavailable")


def advanced_example():
    """
    Advanced example with custom model parameters and multiple predictions
    """
    print("\n" + "=" * 50)
    print("🔬 FloodML Advanced Example")
    print("=" * 50)
    
    # Create predictor with custom Random Forest parameters
    predictor = FloodPredictor(
        usgs_site="01438500",
        flood_stage_ft=25.0,
        model="rf",
        n_estimators=200,        # More trees for better accuracy
        max_depth=15,           # Limit tree depth
        optimize_hyperparameters=True  # Enable hyperparameter tuning
    )
    
    # Train with hyperparameter optimization
    print("🔧 Training with hyperparameter optimization...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=1095)).strftime('%Y-%m-%d')  # 3 years
    
    try:
        predictor.fit(
            start_date=start_date,
            end_date=end_date,
            forecast_hours=24,
            optimize_hyperparameters=True,
            cv_folds=3
        )
        
        # Make predictions for multiple time horizons
        time_horizons = [6, 12, 24, 48, 72]  # hours
        
        print("\n📈 Multi-horizon flood predictions:")
        for hours in time_horizons:
            try:
                prediction = predictor.predict_flood_probability(hours_ahead=hours)
                print(f"   {hours:2d}h: {prediction.probability:6.1%}")
            except:
                print(f"   {hours:2d}h: Unable to predict")
        
        # Get feature importance
        if hasattr(predictor.model, 'get_feature_importance'):
            importance = predictor.model.get_feature_importance()
            if importance is not None:
                print("\n🎯 Top 5 Most Important Features:")
                for i, (feature, score) in enumerate(importance.head().items()):
                    print(f"   {i+1}. {feature}: {score:.3f}")
        
    except Exception as e:
        print(f"❌ Error in advanced example: {e}")


def quick_prediction_example():
    """
    Example using the quick prediction function
    """
    print("\n" + "=" * 50)
    print("⚡ FloodML Quick Prediction")
    print("=" * 50)
    
    try:
        from floodml.prediction.forecaster import quick_flood_prediction
        
        # Quick prediction with minimal setup
        result = quick_flood_prediction(
            usgs_site="01438500",
            flood_stage_ft=25.0,
            hours_ahead=24
        )
        
        print(f"⚡ Quick prediction: {result.probability:.1%} flood probability")
        print(f"🕐 Forecast time: {result.forecast_time}")
        
    except Exception as e:
        print(f"❌ Quick prediction failed: {e}")


def multiple_sites_example():
    """
    Example monitoring multiple USGS sites
    """
    print("\n" + "=" * 50)
    print("🌐 Multi-Site Monitoring")
    print("=" * 50)
    
    # Define multiple high-risk sites
    sites = [
        {"site": "01438500", "name": "Delaware River at Montague, NJ", "flood_stage": 25.0},
        {"site": "01463500", "name": "Delaware River at Trenton, NJ", "flood_stage": 20.0},
        {"site": "02035000", "name": "James River at Richmond, VA", "flood_stage": 18.0},
    ]
    
    print("📊 Monitoring flood risk at multiple sites:")
    
    for site_info in sites:
        try:
            print(f"\n🌊 {site_info['name']}")
            
            predictor = FloodPredictor(
                usgs_site=site_info['site'],
                flood_stage_ft=site_info['flood_stage'],
                model="rf"
            )
            
            # Quick training on 1 year of data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            predictor.fit(start_date, end_date)
            prediction = predictor.predict_flood_probability(hours_ahead=24)
            
            risk_level = "🚨 HIGH" if prediction.probability > 0.7 else "⚠️  MODERATE" if prediction.probability > 0.3 else "✅ LOW"
            print(f"   Risk Level: {risk_level}")
            print(f"   Probability: {prediction.probability:.1%}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")


if __name__ == "__main__":
    # Run all examples
    basic_flood_prediction_example()
    advanced_example()
    quick_prediction_example()
    multiple_sites_example()
    
    print("\n" + "=" * 50)
    print("🎉 FloodML examples completed!")
    print("💡 Modify the examples above to work with your preferred USGS sites")
    print("📖 Check the documentation for more advanced usage patterns")