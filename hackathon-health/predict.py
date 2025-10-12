import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import pickle
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

def categorize_item(item_name):
    item_lower = str(item_name).lower()
    if any(word in item_lower for word in ['vaccine', 'vaccination', 'syringes']):
        return 'vaccine_related'
    elif any(word in item_lower for word in ['antibiotic', 'antifungal', 'topical']):
        return 'antibiotics_wound_care'
    elif any(word in item_lower for word in ['malaria', 'artemether', 'act']):
        return 'antimalarial'
    elif any(word in item_lower for word in ['insulin', 'metformin', 'amlodipine']):
        return 'ncd_medication'
    elif any(word in item_lower for word in ['ors', 'rehydration', 'zinc']):
        return 'rehydration_supplement'
    elif any(word in item_lower for word in ['iv', 'fluid']):
        return 'iv_fluid'
    elif any(word in item_lower for word in ['delivery', 'anc', 'sanitary']):
        return 'maternal_delivery'
    elif any(word in item_lower for word in ['glove', 'mask', 'ppe']):
        return 'ppe_consumables'
    elif any(word in item_lower for word in ['test kit', 'monitor', 'thermometer']):
        return 'diagnostics_triage'
    elif any(word in item_lower for word in ['paracetamol', 'cough syrup']):
        return 'symptomatic_drugs'
    elif any(word in item_lower for word in ['generator', 'record book', 'leaflet']):
        return 'admin_infrastructure'
    else:
        return 'other'

def load_facility_features():
    BASE_DIR = Path(__file__).resolve().parent
    return pd.read_csv(BASE_DIR / "facility_features_precalculated.csv")

def make_prediction(facility_id, item_name, current_stock_level, reorder_level, last_restock_date):
    BASE_DIR = Path(__file__).resolve().parent

    # model loading and other files we used for training the model
    classifier = joblib.load(BASE_DIR / "xgb_classifier.pkl")
    regressor = joblib.load(BASE_DIR / "xgb_regressor.pkl")
    le_category = joblib.load(BASE_DIR / "le_category.pkl")
    available_features = joblib.load(BASE_DIR / "feature_columns.pkl")
    facility_df = load_facility_features()

    # input that the user will see 
    input_data = pd.DataFrame([{
        "facility_id": facility_id,
        "item_name": item_name,
        "stock_level": current_stock_level,
        "reorder_level": reorder_level,
        "last_restock_date": pd.to_datetime(last_restock_date)
    }])

    input_data['days_since_restock'] = (datetime.now() - input_data['last_restock_date']).dt.days
    input_data['restock_frequency'] = np.where(input_data['days_since_restock'] > 0, 365 / input_data['days_since_restock'], 0)
    input_data['stock_status'] = input_data['stock_level'] - input_data['reorder_level']
    input_data['item_category'] = input_data['item_name'].apply(categorize_item)
    input_data['item_category_encoded'] = le_category.transform(input_data['item_category'])

    merged = input_data.merge(facility_df, on='facility_id', how='left')
    if merged.empty:
        raise ValueError("Facility not found in database")

    X = merged[available_features].fillna(0)

    prob = classifier.predict_proba(X)[:, 1][0]
    days = regressor.predict(X)[0]
    risk = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"

    return {
        "probability": f"{prob:.2%}",
        "days_until_stockout": f"{days:.1f}",
        "risk_level": risk
    }

