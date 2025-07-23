import joblib
import pandas as pd
from datetime import datetime

# Define feature columns (same order as used in training)
feature_columns = ['Year', 'Month', 'Day', 'DoW_Friday', 'DoW_Monday', 'DoW_Saturday', 
                  'DoW_Sunday', 'DoW_Thursday', 'DoW_Tuesday', 'DoW_Wednesday']

# Load the trained model
model = joblib.load('../models/gradient_boost.pkl')
def predict_price_for_date(date_input, model):
    """
    Predict natural gas price for any given date.
    
    Parameters:
    date_input: str or datetime - Date in format 'YYYY-MM-DD' or datetime object
    model: trained model (default: best_model from our comparison)
    
    Returns:
    float: predicted price for the given date
    """

    if isinstance(date_input, str):
        date_obj = pd.to_datetime(date_input)
    else:
        date_obj = date_input
    
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day
    day_of_week = date_obj.day_name()
    

    features = {
        'Year': year,
        'Month': month,
        'Day': day,
        'DoW_Friday': 1 if day_of_week == 'Friday' else 0,
        'DoW_Monday': 1 if day_of_week == 'Monday' else 0,
        'DoW_Saturday': 1 if day_of_week == 'Saturday' else 0,
        'DoW_Sunday': 1 if day_of_week == 'Sunday' else 0,
        'DoW_Thursday': 1 if day_of_week == 'Thursday' else 0,
        'DoW_Tuesday': 1 if day_of_week == 'Tuesday' else 0,
        'DoW_Wednesday': 1 if day_of_week == 'Wednesday' else 0
    }
    feature_df = pd.DataFrame([features])[feature_columns]
    
    predicted_price = model.predict(feature_df)[0]
    
    return predicted_price


if __name__ == "__main__":
    print("Natural Gas Price Prediction System")
    print("=" * 40)
    print("Enter a date to get price prediction")
    print("Format: YYYY-MM-DD (e.g., 2025-12-25)")
    print("Type 'quit' to exit")
    print()
    
    while True:
        try:
            # Get user input
            date_input = input("Enter date: ").strip()
            
            # Check if user wants to quit
            if date_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Predict price for the input date
            price = predict_price_for_date(date_input, model)
            date_obj = pd.to_datetime(date_input)
            day_name = date_obj.strftime('%A')
            
            print(f"Date: {date_input} ({day_name})")
            print(f"Predicted Price: ${price:.2f}")
            print("-" * 30)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please enter a valid date in YYYY-MM-DD format")
            print("-" * 30)