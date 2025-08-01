﻿# JPMorgan Quant Research - Financial Risk Analytics & Quantitative Models

A comprehensive quantitative finance project featuring natural gas price prediction, gas storage contract valuation, and credit risk modeling. This project was developed as part of the JPMorgan Chase & Co. Quantitative Research Virtual Internship on Forage.

## 📋 Project Overview

This project consists of four main components:

1. **Task 1**: Natural gas price prediction using machine learning
2. **Task 2**: Gas storage contract pricing and valuation
3. **Task 3**: Credit risk analysis and probability of default modeling
4. **Task 4**: Expected loss calculation and risk management

The analysis covers multiple quantitative finance domains including commodity pricing, storage contract valuation, and credit risk assessment.

### 🎯 Objectives

- Analyze historical natural gas price patterns and trends
- Build robust regression models for price prediction
- Create interactive date-to-price prediction systems
- Develop gas storage contract pricing models
- Implement credit risk assessment and probability of default models
- Calculate expected loss for loan portfolios
- Provide comprehensive risk management and trading insights

## 🔧 Technical Implementation

### Data Analysis

- **Data Source**: Monthly natural gas price snapshots (Oct 2020 - Sep 2024)
- **Features Engineered**: Year, Month, Day, Day-of-Week (one-hot encoded)
- **Exploratory Analysis**: Seasonal patterns, price distributions, temporal trends

### Machine Learning Models

Implemented and compared 5 regression algorithms:

1. **Gradient Boosting Regressor** ⭐ (Best: RMSE 0.28, R² 0.83)
2. Random Forest Regressor (RMSE 0.37, R² 0.72)
3. Lasso Regression (RMSE 0.51, R² 0.47)
4. Ridge Regression (RMSE 0.59, R² 0.28)
5. Linear Regression (RMSE 0.61, R² 0.23)

### Key Features

#### Task 1: Price Prediction

- **Interactive CLI Tool**: Real-time price predictions via command line
- **Date-to-Price Function**: Input any date, get predicted price
- **Future Forecasting**: 12-month price predictions
- **Model Validation**: Cross-validation against historical data
- **Visualization**: Comprehensive charts and trend analysis

#### Task 2: Storage Contract Pricing

- **Contract Valuation**: Calculate storage contract values
- **Constraint Modeling**: Injection/withdrawal rate limits
- **Storage Cost Integration**: Daily storage fees and capacity limits
- **Arbitrage Analysis**: Seasonal price spread opportunities
- **Risk Assessment**: Parameter sensitivity analysis

#### Task 3 & 4: Credit Risk Analysis

- **Probability of Default Modeling**: Machine learning models for credit risk
- **Expected Loss Calculation**: Implementation of PD × LGD × EAD framework
- **Safe Feature Engineering**: Avoiding data leakage in model development
- **Model Comparison**: Performance evaluation across multiple algorithms
- **Risk Categorization**: Low/Medium/High risk classification system

## 📊 Results

### Natural Gas Price Prediction (Task 1)

- **Best Model**: Gradient Boosting Regressor
- **Accuracy**: 83.37% variance explained (R² = 0.8337)
- **Average Error**: ±$0.15 on test predictions
- **Price Range (Next 12 Months)**: $11.43 - $12.75

### Credit Risk Modeling (Tasks 3 & 4)

- **Best Model**: Logistic Regression
- **AUC Score**: 0.778 (realistic performance avoiding data leakage)
- **Key Features**: Income, Years Employed, FICO Score
- **Risk Categories**: Probability-based classification system

## 🚀 Usage

### Installation

```bash
pip install -r requirements.txt
```

### Task 1: Price Prediction

#### Option 1: Interactive Command Line Tool

```bash
python notebooks/task_1.py
```

This launches an interactive session where you can:

- Enter any date in YYYY-MM-DD format
- Get instant price predictions
- See day-of-week information
- Exit by typing 'quit'

Example session:

```
Natural Gas Price Prediction System
========================================
Enter date: 2025-03-15
Date: 2025-03-15 (Saturday)
Predicted Price: $12.55
Enter date: quit
Goodbye!
```

#### Option 2: Jupyter Notebook Analysis

1. Open `notebooks/Task-1.ipynb` in Jupyter Notebook
2. Run all cells to reproduce the full analysis
3. Use the prediction function programmatically:

```python
# Predict price for any date
price = predict_price_for_date('2025-03-15')
print(f"Predicted price: ${price:.2f}")

# Batch predictions
dates = ['2025-01-01', '2025-04-01', '2025-07-01', '2025-10-01']
prices = [predict_price_for_date(date) for date in dates]
```

### Task 2: Gas Storage Contract Pricing

```bash
python notebooks/task-2.py
```

Use the gas storage contract pricing function:

```python
from task2 import price_gas_storage_contract, load_price_curve

# Load price data
price_curve = load_price_curve("data/raw/Nat_Gas.csv")

# Define injection and withdrawal schedules
injection_schedule = [("2022-05-01", 50), ("2022-05-02", 50)]
withdrawal_schedule = [("2022-06-01", 60), ("2022-06-10", 40)]

# Calculate contract value
value = price_gas_storage_contract(
    injection_schedule,
    withdrawal_schedule,
    price_curve,
    injection_rate=60,        # Max daily injection rate
    withdrawal_rate=60,       # Max daily withdrawal rate
    max_storage=100,          # Storage capacity
    storage_cost_per_day=0.01 # Daily storage cost per unit
)

print(f"Contract Value: ${value}")
```

### Task 3 & 4: Credit Risk Analysis

```bash
# Open the Jupyter notebook
jupyter notebook notebooks/task_3_4.ipynb
```

Use the expected loss function for credit risk assessment:

```python
from task_3_4 import predict_loan_expected_loss

# Calculate expected loss for a loan application
result = predict_loan_expected_loss(
    income=75000,           # Annual income
    years_employed=5,       # Years of employment
    fico_score=720,         # FICO credit score
    loan_amount=25000,      # Loan amount
    recovery_rate=0.10      # Expected recovery rate (10%)
)

print(f"Default Probability: {result['probability_default']:.2%}")
print(f"Expected Loss: ${result['expected_loss']:,.2f}")
print(f"Risk Category: {result['risk_category']}")
```

## 📁 Project Structure

```
jpmorgan-quant-forage/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── notebooks/
│   ├── Task-1.ipynb         # Natural gas price prediction analysis
│   ├── task_1.py            # Interactive price prediction tool
│   ├── task-2.py            # Gas storage contract pricing model
│   └── task_3_4.ipynb       # Credit risk analysis and expected loss modeling
├── data/
│   ├── raw/                 # Original datasets
│   │   ├── Nat_Gas.csv      # Natural gas prices data
│   │   └── Task 3 and 4_Loan_Data.csv  # Credit risk dataset
│   └── cleaned/             # Processed data (generated)
├── models/                  # Trained models (generated)
│   ├── gradient_boost.pkl   # Gas price prediction model
│   ├── credit_risk_model.pkl # Credit risk model
│   └── credit_risk_scaler.pkl # Feature scaler for credit model
└── LICENSE                  # Project license
```

## 🛠️ Technologies Used

- **Python 3.8+**
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, Imbalanced-learn (SMOTE)
- **Model Persistence**: Joblib
- **Interactive Tools**: Command-line interfaces
- **Model Persistence**: Joblib
- **Development Environment**: Jupyter Notebook

## 💼 Business Applications

### Natural Gas Trading

- **Price Volatility Assessment**: Quantify price movement risks
- **Hedging Strategies**: Inform derivative pricing models
- **Portfolio Optimization**: Balance gas storage positions

### Storage Contract Valuation

- **Market Timing**: Identify optimal buy/sell periods
- **Arbitrage Opportunities**: Spot pricing inefficiencies
- **Contract Valuation**: Price long-term storage agreements

### Credit Risk Management

- **Loan Underwriting**: Automated risk assessment for loan applications
- **Portfolio Management**: Expected loss calculation for loan portfolios
- **Capital Allocation**: Risk-adjusted pricing and provisioning
- **Regulatory Compliance**: Basel framework implementation (PD, LGD, EAD)

## 🔍 Key Insights

### Natural Gas Market Analysis

1. **Seasonal Patterns**: Clear winter (Dec-Feb) price premiums
2. **Volatility Trends**: Higher price variance in heating months
3. **Day-of-Week Effects**: Minimal but measurable weekday patterns
4. **Predictability**: Strong temporal correlations enable accurate forecasting

### Credit Risk Findings

1. **Data Leakage Risk**: High AUC scores (>99%) often indicate post-loan features
2. **Realistic Performance**: Safe models achieve 65-80% AUC using pre-loan features
3. **Feature Importance**: Income, employment history, and FICO score are key predictors
4. **Risk Distribution**: Clear separation between low/medium/high risk borrowers

## 📝 Future Enhancements

### Natural Gas Models

- [ ] Incorporate external factors (weather, supply data)
- [ ] Implement time series models (ARIMA, LSTM)
- [ ] Add confidence intervals to predictions
- [ ] Real-time data integration

### Credit Risk Models

- [ ] Implement survival analysis for time-to-default
- [ ] Add macroeconomic stress testing
- [ ] Develop dynamic risk scoring
- [ ] Create portfolio optimization models

### General Improvements

- [ ] Web-based prediction interfaces
- [ ] API development for model deployment
- [ ] Advanced visualization dashboards
- [ ] Model monitoring and drift detection

## 👨‍💼 About

This project demonstrates comprehensive quantitative analysis skills relevant to:

- **Quantitative Research**: Statistical modeling, hypothesis testing, and financial analytics
- **Risk Management**: Price forecasting, volatility analysis, and credit risk assessment
- **Algorithmic Trading**: Systematic strategy development and market analysis
- **Data Science**: End-to-end ML pipeline implementation with production-ready code
- **Financial Engineering**: Derivative pricing, contract valuation, and portfolio optimization

**Skills Demonstrated:**

- Machine Learning model development and validation
- Time series analysis and forecasting
- Credit risk modeling with regulatory framework compliance
- Data preprocessing and feature engineering
- Model interpretation and business insight generation
- Production code development with error handling and documentation

---

**Author**: Quantum-coderX
**Program**: JPMorgan Chase Quantitative Research Virtual Internship  
**Platform**: Forage  
**Completion Date**: august 2025

