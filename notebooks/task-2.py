import pandas as pd
from datetime import datetime
from typing import List, Tuple, Dict

def load_price_curve(csv_path: str) -> Dict[datetime, float]:
    """
    Reads a CSV with columns Dates, Prices and returns a dictionary {date: price}.
    """
    df = pd.read_csv(csv_path)
    # Handle the actual column names from the CSV
    if 'Dates' in df.columns and 'Prices' in df.columns:
        df['Date'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
        df['Price'] = df['Prices']
    else:
        df['Date'] = pd.to_datetime(df['Date'])
    
    df = df.set_index('Date').sort_index()
    return df['Price'].to_dict()

def price_gas_storage_contract(
    injection_schedule: List[Tuple[str, float]],
    withdrawal_schedule: List[Tuple[str, float]],
    price_curve: Dict[datetime, float],
    injection_rate: float,
    withdrawal_rate: float,
    max_storage: float,
    storage_cost_per_day: float
) -> float:
    """
    Calculates the value of a gas storage contract based on injection/withdrawal schedule and constraints.
    """
    inject_map = {pd.to_datetime(date): vol for date, vol in injection_schedule}
    withdraw_map = {pd.to_datetime(date): vol for date, vol in withdrawal_schedule}
    
    all_dates = sorted(set(price_curve.keys()).union(inject_map.keys()).union(withdraw_map.keys()))
    
    storage_level = 0.0
    total_value = 0.0
    total_storage_cost = 0.0

    for date in all_dates:
        price = price_curve.get(date, 0.0)

        # Inject
        injected = inject_map.get(date, 0.0)
        injected = min(injected, injection_rate)
        injected = min(injected, max_storage - storage_level)
        storage_level += injected
        total_value -= injected * price

        # Withdraw
        withdrawn = withdraw_map.get(date, 0.0)
        withdrawn = min(withdrawn, withdrawal_rate, storage_level)
        storage_level -= withdrawn
        total_value += withdrawn * price

        # Daily storage cost
        total_storage_cost += storage_level * storage_cost_per_day

    return round(total_value - total_storage_cost, 2)

# Test block (can be removed if using as a module)
if __name__ == "__main__":
    # Use the correct path to the CSV file
    price_curve = load_price_curve("../data/raw/Nat_Gas.csv")
    
    injection_schedule = [("2022-05-01", 50), ("2022-05-02", 50)]
    withdrawal_schedule = [("2022-06-01", 60), ("2022-06-10", 40)]
    
    value = price_gas_storage_contract(
        injection_schedule,
        withdrawal_schedule,
        price_curve,
        injection_rate=60,
        withdrawal_rate=60,
        max_storage=100,
        storage_cost_per_day=0.01
    )
    
    print("Contract Value:", value)
