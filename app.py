# app.py
# Streamlit app for predicting house prices in Ireland using ordinal encoding.

import streamlit as st
import pandas as pd
import joblib

# Load trained models
clf = joblib.load('classification_model.pkl')
reg = joblib.load('regression_model.pkl')

# Category mappings
property_type_order = ['Detached', 'Duplex', 'Townhouse', 'End Of Terrace', 'Semi-D', 'Terrace', 'Bungalow', 'House', 'Apartment', 'Studio']
ber_rating_order = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'D1', 'D2', 'E1', 'E2', 'F', 'G']
county_order = [
    'Dublin', 'Cork', 'Galway', 'Limerick', 'Waterford', 'Mayo', 'Meath', 'Kildare', 'Wicklow', 'Other'
]  # Replace 'Other' with actual values used in training if needed

category_map = {
    0: 'High',
    1: 'Medium-High',
    2: 'Medium',
    3: 'Medium-Low',
    4: 'Low'
}

def main():
    st.title("🏠 Ireland House Price Predictor")

    tab1, tab2 = st.tabs(["📊 Predict Price Category", "💶 Predict Price Value"])

    with tab1:
        st.header("Classification - Price Category")
        user_input = get_user_input(prefix="cat")
        if st.button("Predict Price Category"):
            pred_cat = clf.predict(user_input)
            label = category_map.get(pred_cat[0], "Unknown")
            st.success(f"Predicted Category: **{label}**")

    with tab2:
        st.header("Regression - Predict Price (€)")
        user_input = get_user_input(prefix="val")
        if st.button("Predict Price Value"):
            pred_price = reg.predict(user_input)
            st.success(f"Estimated Price: **€{int(pred_price[0]):,}**")

def get_user_input(prefix=""):
    st.subheader("Enter Property Details")

    size = st.number_input("Floor Area (m²)", min_value=20, max_value=500, value=100, key=f"{prefix}_size")
    beds = st.slider("Number of Bedrooms", 1, 10, 3, key=f"{prefix}_beds")
    baths = st.slider("Number of Bathrooms", 1, 10, 2, key=f"{prefix}_baths")
    property_type = st.selectbox("Property Type", property_type_order, key=f"{prefix}_ptype")
    county = st.selectbox("County", county_order, key=f"{prefix}_county")
    ber = st.selectbox("BER Rating", ber_rating_order, key=f"{prefix}_ber")
    construction = st.number_input("Date of Construction", min_value=1800, max_value=2025, value=2000, key=f"{prefix}_date")

    # Build initial data dictionary
    data = {
        'Floor Area (m2)': size,
        'Number of Bedrooms': beds,
        'Number of Bathrooms': baths,
        'Property Type': property_type_order.index(property_type),
        'County': county_order.index(county) if county in county_order else -1,
        'BER Rating': ber_rating_order.index(ber) if ber in ber_rating_order else -1,
        'Date of Construction': construction,
        'Price_per_m2': 0  # Dummy placeholder; model might use this
    }

    # Detect the required feature order from the classifier or regressor
    model_features = clf.feature_names_in_ if hasattr(clf, 'feature_names_in_') else list(data.keys())

    # Create the input dataframe in the correct order
    input_df = pd.DataFrame([[data.get(col, 0) for col in model_features]], columns=model_features)

    return input_df



if __name__ == "__main__":
    main()