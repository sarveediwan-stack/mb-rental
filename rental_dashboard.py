import streamlit as st
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import seaborn as sns
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from geopy.distance import geodesic
import os
import io
from datetime import datetime

if 'last_selected_society' not in st.session_state:
    st.session_state.last_selected_society = None
if 'last_selected_bhk' not in st.session_state:
    st.session_state.last_selected_bhk = None

# Set page configuration
st.set_page_config(
    page_title="Real Estate Rental Analysis Dashboard",
    page_icon="🏠",
    layout="wide"
)

# App title and description
st.title("🏠 Magicbricks Rental Analysis Dashboard")
st.markdown("""
This app analyzes rental property data to predict rents and find comparable properties.
""")

# # Sidebar for file upload
# with st.sidebar:
#     st.header("Data Input")
#     uploaded_file = st.file_uploader("Upload CSV data file", type=["csv"])

# Function to check if data directory exists
def ensure_data_dir():
    os.makedirs("data", exist_ok=True)

# Function to load and preprocess data
def load_and_process_data():
    # Use hardcoded file path instead of uploaded_file
    file_path = "data/magicbricks_hyderabad_properties_compressed.csv"  # This file should be in your GitHub repo
    
    try:
        df_raw = pd.read_csv(file_path)
        # Rename and normalize column names
        df_raw.columns = df_raw.columns.str.strip().str.lower()
        
        # Define column mapping
        rename_map = {
            'property_localityname': 'locality',
            'detail_propertytype': 'property_type',
            'detail_coveredarea': 'builtup_area',
            'detail_carpetarea': 'carpet_area',
            'detail_bedrooms': 'bedrooms',
            'detail_bathrooms': 'bathrooms',
            'detail_furnished': 'furnishing',
            'detail_floornumber': 'floor',
            'detail_totalfloornumber': 'total_floors',
            'detail_ageofcons': 'building_age',
            'detail_numberofbalconied': 'balconies',
            'detail_projectname': 'society',
            'detail_maintenancecharges': 'maintenance',
            'detail_exactsalerentprice': 'rent',
            'detail_latitude': 'latitude',
            'detail_longitude': 'longitude',
            'detail_facing': 'facing',
            'detail_waterstatus': 'water_status',
            'detail_carparking': 'car_parking',
            'detail_electricitystatus': 'electricity_status',
            'detail_pricebreakup_securitydeposit': 'deposit',
            'detail_coveredareasqft': 'rent_per_sqft',
            'detail_additionalrooms': 'detail_additionalrooms'
        }
        
        df = df_raw.rename(columns=rename_map)
        # Drop rows with missing values in critical fields
        critical_fields = ['rent', 'bedrooms']
        critical_fields = [col for col in critical_fields if col in df.columns]
        if critical_fields:
            df = df.dropna(subset=critical_fields)

        # Convert numeric columns
        numeric_cols = {
            'rent': 0,
            'builtup_area': 0,
            'carpet_area': 0,
            'bedrooms': 0,
            'bathrooms': 0,
            'floor': 1,
            'total_floors': 1
        }
        for col, default in numeric_cols.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df['floor'] = df['floor'].fillna(1)
        df['total_floors'] = df['total_floors'].fillna(1)

        # Remove outliers
        def remove_outliers(df, column):
            if column not in df.columns:
                return df
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
        # for col in ['rent', 'builtup_area', 'total_floors']:
        for col in ['rent']:
            if col in df.columns:
                df = remove_outliers(df, col)
        # Extract amenity columns
        amenity_cols = [col for col in df.columns if col.startswith('detail_amenitymap_') or col.startswith('detail_amenityexternalmap_')]
        amenity_df = df[amenity_cols].copy()
        amenity_df.columns = [col.replace('detail_amenitymap_', '').replace('detail_amenityexternalmap_', '') for col in amenity_cols]
        amenity_df = amenity_df.fillna(0).applymap(lambda x: 1 if str(x).strip().lower() in ['true', '1', 'yes', 'present'] else 0)

        # Concatenate back to main df
        df = pd.concat([df, amenity_df], axis=1)
        # Filter for necessary columns
        required_cols = [
            'locality', 'society', 'property_type', 'builtup_area', 'carpet_area','bedrooms', 'bathrooms', 'furnishing',
            'floor', 'total_floors', 'rent', 'latitude', 'longitude', 'detail_additionalrooms', 'building_age','maintenance'
        ]
        existing_cols = [col for col in required_cols if col in df.columns]
        df = df[existing_cols].copy() if existing_cols else df
        
        # Filter for multi-storey apartments if column exists
        if 'property_type' in df.columns:
            df = df[df['property_type'].str.lower().str.strip() == 'multistorey apartment']
        
        # Impute Missing Values
        df['builtup_area_missing'] = df['builtup_area'].isna().astype(int)
        if 'carpet_area' in df.columns:
            # Create a mask for rows with missing builtup_area but available carpet_area
            mask = df['builtup_area'].isna() & df['carpet_area'].notna()
            
            # For these rows, calculate builtup_area as 1.1 * carpet_area
            df.loc[mask, 'builtup_area'] = df.loc[mask, 'carpet_area'] * 1.1
            
            # st.success(f"Filled {mask.sum()} missing builtup_area values using carpet_area")
        
        # Drop rows where both builtup_area and carpet_area are missing
        if 'carpet_area' in df.columns:
            missing_both = (df['builtup_area'].isna() & df['carpet_area'].isna())
            
            rows_to_drop = missing_both.sum()
            # st.success(f"Dropping {rows_to_drop} rows with both builtup_area and carpet_area missing")
            
            df = df[~missing_both]
        else:
            # If carpet_area column doesn't exist, just drop rows with missing builtup_area
            rows_to_drop = df['builtup_area'].isna().sum()
            # st.success(f"Dropping {rows_to_drop} rows with missing builtup_area (carpet_area column not found)")
            
            df = df.dropna(subset=['builtup_area'])
        # st.success(f"✅ Data AFTER AREA CLEAN successfully: {len(df)} properties")
        df['locality_missing'] = df['locality'].isna().astype(int)
        df['society_missing'] = df['society'].isna().astype(int)
        df['lat_missing'] = df['latitude'].isna().astype(int)
        df['lng_missing'] = df['longitude'].isna().astype(int)
        # Then perform your imputation
        df['locality'] = df['locality'].fillna('Unknown')
        df['society'] = df['society'].fillna('Unknown')
        df['latitude'] = df['latitude'].fillna(df['latitude'].median())
        df['longitude'] = df['longitude'].fillna(df['longitude'].median())
        
        # Process additional rooms if present
        if 'detail_additionalrooms' in df.columns:
            df['detail_additionalrooms'] = df['detail_additionalrooms'].astype(str).fillna('').str.lower()
            df['has_study'] = df['detail_additionalrooms'].str.contains('study').astype(int)
            if 'bedrooms' in df.columns:
                df['bedrooms'] = df['bedrooms'] + 0.5 * df['has_study']
        
        # Calculate total rent
        if 'maintenance' not in df.columns:
            df['maintenance'] = 0
        else:
            # df['maintenance'] = pd.to_numeric(df['maintenance'], errors='coerce').fillna(0)
            df['maintenance'] = df['maintenance'].astype(str).str.replace(',', '')
            # Then convert to numeric
            df['maintenance'] = pd.to_numeric(df['maintenance'], errors='coerce').fillna(0)




        
        if 'rent' in df.columns:
            df['total_rent'] = df['rent'] + df['maintenance']
        
        # Process building age if present
        if 'building_age' in df.columns:
            age_map = {
                'New Construction': 1,
                'Under Construction': 0,
                'Less than 5 years': 3,
                '5 to 10 years': 7,
                '10 to 15 years': 13,
                '15 to 20 years': 18,
                'Above 20 years': 25
            }
            df['building_age'] = df['building_age'].map(age_map)
            # Fill missing values
            if 'society' in df.columns:
                df['building_age'] = df.groupby('society')['building_age'].transform(lambda x: x.fillna(x.median()))
            if 'locality' in df.columns:
                df['building_age'] = df.groupby('locality')['building_age'].transform(lambda x: x.fillna(x.median()))
            df['building_age'] = df['building_age'].fillna(df['building_age'].median() if not df['building_age'].isna().all() else 0)
        
        
        # Create log transformations
        if 'total_rent' in df.columns:
            df['log_total_rent'] = np.log1p(df['total_rent'])
        if 'rent' in df.columns:
            df['log_rent'] = np.log1p(df['rent'])
        if 'builtup_area' in df.columns:
            df['log_builtup_area'] = np.log1p(df['builtup_area'])
        
        # Handle zero or missing values
        if 'floor' in df.columns:
            df['floor'] = df['floor'].fillna(1).replace(0, 1)
        if 'total_floors' in df.columns:
            df['total_floors'] = df['total_floors'].fillna(1).replace(0, 1)
        
        # Create feature
        if 'floor' in df.columns and 'total_floors' in df.columns:
            df['floor_to_total_floors'] = df['floor'] / df['total_floors']
        
        # Filter to keep reasonable values
        if 'bedrooms' in df.columns:
            df = df[(df['bedrooms'] >= 1) & (df['bedrooms'] <= 7)]
        if 'bathrooms' in df.columns:
            df = df[(df['bathrooms'] >= 1) & (df['bathrooms'] <= 7)]
        
        # Encode categorical variables
        label_encoders = {}
        categoricals = ['locality', 'society', 'furnishing']
        
        for col in categoricals:
            if col in df.columns:
                df[col] = df[col].astype(str)
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le
        
        # Save the processed data
        ensure_data_dir()
        df.to_csv('data/processed_data.csv', index=False)
        # st.success(f"✅ Data AFTER ALL DONE successfully: {len(df)} properties")
        return df, label_encoders
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


def build_society_locality_map(df, label_encoders):
    """
    Creates a mapping from society names to their canonical locality names.
    Works with the original string values, not encoded values.
    
    Args:
        df: DataFrame with encoded society and locality values
        label_encoders: Dictionary of label encoders for categorical variables
        
    Returns:
        Dictionary mapping society names (strings) to canonical locality names (strings)
    """
    # Create inverse mappings to get original string values
    society_encoder = label_encoders['society']
    locality_encoder = label_encoders['locality']
    
    # Decode the encoded values to get original strings
    society_mapping = {i: name for i, name in enumerate(society_encoder.classes_)}
    locality_mapping = {i: name for i, name in enumerate(locality_encoder.classes_)}
    
    # Create a DataFrame with decoded values for analysis
    decoded_df = df.copy()
    decoded_df['society_name'] = decoded_df['society'].map(society_mapping)
    decoded_df['locality_name'] = decoded_df['locality'].map(locality_mapping)
    
    # Build the canonical mapping
    society_locality_map = {}
    
    # Group by society name and find most common locality
    society_groups = decoded_df.groupby('society_name')
    
    for society, group in society_groups:
        if society == 'Unknown' or pd.isna(society):
            continue
            
        # Find most frequent locality for this society
        locality_counts = group['locality_name'].value_counts()
        if len(locality_counts) > 0:
            canonical_locality = locality_counts.index[0]
            society_locality_map[society] = canonical_locality
            
    return society_locality_map


def apply_canonical_locality_to_dataset(df, label_encoders):
    """
    Applies the canonical locality mapping to the entire dataset before training.
    This ensures consistent society-locality relationships throughout the model.
    
    Args:
        df: DataFrame with encoded values
        label_encoders: Dictionary of label encoders
        
    Returns:
        DataFrame with consistent locality values
    """
    print("Applying canonical locality mapping to training data...")
    
    # First, build the society-locality map
    society_locality_map = build_society_locality_map(df, label_encoders)
    
    # Create a mapping dictionary from encoded society to encoded canonical locality
    society_to_canonical_locality_encoded = {}
    
    # For each society, get its canonical locality and encode both
    for society_name, canonical_locality_name in society_locality_map.items():
        try:
            # Get encoded values
            society_encoded = label_encoders['society'].transform([society_name])[0]
            canonical_locality_encoded = label_encoders['locality'].transform([canonical_locality_name])[0]
            
            # Map encoded society to encoded canonical locality
            society_to_canonical_locality_encoded[society_encoded] = canonical_locality_encoded
        except:
            # Skip if encoding fails
            continue
    
    # Count how many rows will be modified
    modified_count = 0
    
    # Create a copy of the DataFrame to modify
    df_consistent = df.copy()
    
    # For each row in the dataset
    for idx, row in df_consistent.iterrows():
        society_encoded = row['society']
        locality_encoded = row['locality']
        
        # If this society has a canonical locality mapping
        if society_encoded in society_to_canonical_locality_encoded:
            canonical_locality_encoded = society_to_canonical_locality_encoded[society_encoded]
            
            # If current locality is different from canonical locality
            if locality_encoded != canonical_locality_encoded:
                # Update locality to canonical locality
                df_consistent.at[idx, 'locality'] = canonical_locality_encoded
                modified_count += 1
    
    print(f"Modified {modified_count} rows ({modified_count/len(df)*100:.2f}%) to use canonical localities")
    
    # Return the updated DataFrame and the mapping for future use
    return df_consistent, society_locality_map



def predict_rent_with_canonical_locality(input_data, society_locality_map, models, label_encoders):
    """
    Ensures consistent locality is used for a given society before prediction.
    
    Args:
        input_data: Dictionary with property details
        society_locality_map: Dictionary mapping society names to canonical localities
        
    Returns:
        Dictionary with prediction results
    """
    if models is None:
        return {'model_a_raw_prediction': 0, 'model_b_log_prediction': 0}
        
    # Create a copy to avoid modifying the original
    input_copy = input_data.copy()
    
    # Get society and locality from input
    society = input_copy.get('society')
    current_locality = input_copy.get('locality')
    
    # Check if this society has a canonical locality mapping
    if society in society_locality_map:
        canonical_locality = society_locality_map[society]
        
        # Only update if different from current
        if current_locality != canonical_locality:
            print(f"Note: For consistency, using canonical locality '{canonical_locality}' for society '{society}' (was '{current_locality}')")
            input_copy['locality'] = canonical_locality
    
    # Now make the prediction with the adjusted input
    return predict_rent_dual(input_copy, models, label_encoders)

# Function to train prediction models
def train_models(df):
    if df is None:
        return None
    df['log_builtup_area'] = np.log1p(df['builtup_area'])
    features = ['bedrooms', 'builtup_area', 'bathrooms', 'furnishing', 'locality', 
               'society', 'floor', 'total_floors', 'building_age']
    
    # Make sure all features exist
    features = [f for f in features if f in df.columns]
    
    if not features or 'total_rent' not in df.columns:
        st.error("Missing required columns for model training")
        return None
    
    try:
        # Update features list to use log_builtup_area instead of builtup_area
        features_with_log = [f if f != 'builtup_area' else 'log_builtup_area' for f in features]
        
        # Train models with updated features
        # X = df[features_with_log]
        # # X = df[features]
        # y_a = df['total_rent']
        # y_b = df['log_total_rent']

        # Apply canonical locality mapping to ensure consistent training data
        df_consistent, society_locality_map = apply_canonical_locality_to_dataset(df, label_encoders)
        
        # Train models with updated features and consistent data
        X = df_consistent[features_with_log]
        y_a = df_consistent['total_rent']       # Model A: raw rent
        y_b = df_consistent['log_total_rent']   # Model B: log-transformed rent
        X_train, X_test, y_train_a, y_test_a = train_test_split(X, y_a, test_size=0.2, random_state=42)
        _, _, y_train_b, y_test_b = train_test_split(X, y_b, test_size=0.2, random_state=42)
        
        # Model A - No log
        model_a = RandomForestRegressor(n_estimators=100, random_state=42)
        model_a.fit(X_train, y_train_a)
        
        # Model B - Log-transformed
        model_b = RandomForestRegressor(n_estimators=100, random_state=42)
        model_b.fit(X_train, y_train_b)
        
        # Evaluate models
        y_pred_a = model_a.predict(X_test)
        log_preds_b = model_b.predict(X_test)
        y_pred_b = np.expm1(log_preds_b)
        
        mae_a = mean_absolute_error(y_test_a, y_pred_a)
        rmse_a = np.sqrt(mean_squared_error(y_test_a, y_pred_a))
        
        mae_b = mean_absolute_error(y_test_a, y_pred_b)
        rmse_b = np.sqrt(mean_squared_error(y_test_a, y_pred_b))
        
        # Return models and evaluation metrics
        return {
            'model_a': model_a,
            'model_b': model_b,
            'features': features,
            'mae_a': mae_a,
            'rmse_a': rmse_a,
            'mae_b': mae_b,
            'rmse_b': rmse_b
        }
    except Exception as e:
        st.error(f"Error training models: {e}")
        return None

# Function to predict rent
def predict_rent_dual(input_data, models, label_encoders):
    if models is None:
        return {'model_a_raw_prediction': 0, 'model_b_log_prediction': 0}
    
    input_df = pd.DataFrame([input_data]).copy()
    
    # Feature engineering
    input_df['floor'] = input_df['floor'].replace(0, 1)
    input_df['total_floors'] = input_df['total_floors'].replace(0, 1)
    
    if 'floor' in input_df.columns and 'total_floors' in input_df.columns:
        input_df['floor_to_total_floors'] = input_df['floor'] / input_df['total_floors']

    # Add log transformation for builtup_area
    input_df['log_builtup_area'] = np.log1p(input_df['builtup_area'])

    # Encode categorical variables
    for col in ['furnishing', 'locality', 'society']:
        try:
            if col in input_df.columns and col in label_encoders:
                input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
        except ValueError as e:
            st.error(f"Value '{input_df[col][0]}' not found in training data for column {col}")
            return {
                'model_a_raw_prediction': 0,
                'model_b_log_prediction': 0
            }


    # Make predictions
    try:
        features = ['bedrooms', 'builtup_area', 'bathrooms', 'furnishing', 'locality', 
               'society', 'floor', 'total_floors', 'building_age']

        # Select model features
        features_with_log = [f if f != 'builtup_area' else 'log_builtup_area' for f in features]
        input_features = input_df[features_with_log]

        # input_features = input_df[models['features']]
        
        # Predict using Model A (raw rent)
        pred_a = models['model_a'].predict(input_features)[0]
        
        # Predict using Model B (log rent), then reverse log
        pred_log_b = models['model_b'].predict(input_features)[0]
        pred_b = np.expm1(pred_log_b)
        
        return {
            'model_a_raw_prediction': pred_a,
            'model_b_log_prediction': pred_b
        }
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return {
            'model_a_raw_prediction': 0,
            'model_b_log_prediction': 0
        }

# Function to find comparable properties
def find_comparables(df, label_encoders, society, bhk, lat, lon, radius_km=2.0):
    try:
        # Encode society
        society_encoded = label_encoders['society'].transform([society])[0] if 'society' in label_encoders else -1
        
        # Filter for properties
        same_society = df[df['society'] == society_encoded] if society_encoded != -1 else pd.DataFrame()
        same_bhk = df[(df['bedrooms'] == bhk) & (df['society'] != society_encoded)] if society_encoded != -1 and 'bedrooms' in df.columns else pd.DataFrame()
        
        # Calculate distances
        df_temp = df.copy()
        if 'latitude' in df_temp.columns and 'longitude' in df_temp.columns:
            df_temp['distance_km'] = df_temp.apply(
                lambda row: geodesic((lat, lon), (row['latitude'], row['longitude'])).km 
                if not pd.isna(row['latitude']) and not pd.isna(row['longitude']) else float('inf'), 
                axis=1
            )
            nearby = df_temp[df_temp['distance_km'] <= radius_km]
        else:
            nearby = pd.DataFrame()
        
        return same_society, same_bhk, nearby
    except Exception as e:
        st.error(f"Error finding comparable properties: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Function to calculate base rent per square foot
def get_base_rent_per_sqft(df, label_encoders, locality, society):
    try:
        # Use most common furnishing type as base
        if 'furnishing' in df.columns:
            base_df = df[(df['furnishing'] == df['furnishing'].mode()[0])]
        else:
            base_df = df.copy()
        
        # Encode the inputs
        locality_encoded = label_encoders['locality'].transform([locality])[0] if 'locality' in label_encoders else -1
        society_encoded = label_encoders['society'].transform([society])[0] if 'society' in label_encoders else -1
        
        # Filter using encoded values
        soc_df = base_df[base_df['society'] == society_encoded] if society_encoded != -1 else pd.DataFrame()
        loc_df = base_df[base_df['locality'] == locality_encoded] if locality_encoded != -1 else pd.DataFrame()
        
        # Calculate rent per square foot
        soc_rent_psf = (soc_df['rent'].sum() / soc_df['builtup_area'].sum() 
                        if not soc_df.empty and 'rent' in soc_df.columns 
                        and 'builtup_area' in soc_df.columns 
                        and soc_df['builtup_area'].sum() > 0 else 0)
        
        loc_rent_psf = (loc_df['rent'].sum() / loc_df['builtup_area'].sum() 
                        if not loc_df.empty and 'rent' in loc_df.columns 
                        and 'builtup_area' in loc_df.columns 
                        and loc_df['builtup_area'].sum() > 0 else 0)
        
        return soc_rent_psf if soc_rent_psf > 0 else loc_rent_psf
    except Exception as e:
        st.error(f"Error calculating rent per square foot: {e}")
        return 0

# Function to adjust rent for furnishing
def adjust_rent_for_furnishing(base_rent, furnishing):
    if furnishing == 'Furnished':
        return base_rent * 1.1
    elif furnishing == 'UnFurnished':
        return base_rent * 0.95
    return base_rent

# Function to estimate rent using alternative method
def estimate_rent_alternative(df, label_encoders, area, locality, society, furnishing):
    try:
        base_rent_psf = get_base_rent_per_sqft(df, label_encoders, locality, society)
        base_rent = base_rent_psf * area
        adjusted_rent = adjust_rent_for_furnishing(base_rent, furnishing)
        return adjusted_rent
    except Exception as e:
        st.error(f"Error estimating alternative rent: {e}")
        return 0

# Function to generate waterfall
def generate_shap_waterfall(model, features, input_data, feature_names, label_encoders):
    """
    Generate a SHAP waterfall plot with properly formatted labels
    """
    # Create a SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for the input property
    shap_values = explainer(input_data)
    
    # Create a dictionary to map encoded values back to original names
    decoded_feature_names = feature_names.copy()
    
    # Decode the categorical features for display
    for col in ['furnishing', 'locality', 'society']:
        if col in input_data.columns and col in label_encoders:
            # Get the original value
            encoded_value = input_data[col].values[0]
            original_value = label_encoders[col].inverse_transform([int(encoded_value)])[0]
            # Update the display name
            idx = list(input_data.columns).index(col)
            decoded_feature_names[idx] = f"{col}: {original_value}"
    
    # Create waterfall plot
    fig = plt.figure(figsize=(10, 8))
    max_features_to_show = min(len(features), 10)  # Limit to top 10 features
    shap_values.feature_names = decoded_feature_names
    shap.plots.waterfall(shap_values[0], max_display=max_features_to_show, show=False)
    
    # Add title
    plt.title("Feature Contribution to Rent Prediction", fontsize=16)
    plt.tight_layout()
    
    return fig

# Main app logic 
df, label_encoders = load_and_process_data()

if df is not None and len(df) > 0:
    st.success(f"✅ Data loaded successfully: {len(df)} properties")
    
    # Train the models
    with st.spinner("Training models..."):
        models = train_models(df)
    
    if models is not None:
        st.success("✅ Models trained successfully!")
        
        # Create tabs for different functionality
        tabs = st.tabs(["Rent Prediction", "Comparable Properties", "Data Exploration","Landlord Report"])
        
        # Rent Prediction Tab
        with tabs[0]:
            st.header("Rent Prediction")
            with st.form(key="rent_prediction_form"):
    
                
                # Input columns layout
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Get unique societies and localities
                    unique_societies = list(label_encoders['society'].classes_) if 'society' in label_encoders else []
                    unique_localities = list(label_encoders['locality'].classes_) if 'locality' in label_encoders else []
                    
                    # Property inputs
                    selected_locality = st.selectbox("Locality", unique_localities)
                    selected_society = st.selectbox("Society", unique_societies)
                    
                with col2:
                    selected_bhk = st.number_input("Bedrooms", min_value=1.0, max_value=7.0, value=3.0, step=0.5)
                    selected_bathrooms = st.number_input("Bathrooms", min_value=1, max_value=7, value=2)
                    selected_area = st.number_input("Built-up Area (sqft)", min_value=100, max_value=10000, value=1500)
                    
                with col3:
                    furnishing_options = list(label_encoders['furnishing'].classes_) if 'furnishing' in label_encoders else []
                    selected_furnishing = st.selectbox("Furnishing", furnishing_options)
                    selected_floor = st.number_input("Floor", min_value=1, max_value=50, value=5)
                    selected_total_floors = st.number_input("Total Floors", min_value=1, max_value=50, value=10)
                    
                # Optional fields
                with st.expander("Additional Fields (Optional)"):
                    building_age = st.slider("Building Age (years)", min_value=0, max_value=30, value=5)
        
                # Submit button - THIS IS THE ONLY BUTTON NEEDED
                predict_submitted = st.form_submit_button("Predict Rent")

            if predict_submitted:
                # Create input property dictionary
                input_property = {
                    'bedrooms': selected_bhk,
                    'builtup_area': selected_area,
                    'bathrooms': selected_bathrooms,
                    'furnishing': selected_furnishing,
                    'locality': selected_locality,
                    'society': selected_society,
                    'floor': selected_floor,
                    'total_floors': selected_total_floors,
                    'building_age': building_age
                }
                
                # Predict button
                # if st.button("Predict Rent", key="predict_button"):
                with st.spinner("Predicting..."):
                    # ML-based Prediction
                    # Step 1: Create the society-locality mapping
                    society_locality_map = build_society_locality_map(df, label_encoders)
                    # Step 2: Predict
                    results = predict_rent_with_canonical_locality(input_property, society_locality_map, models, label_encoders)
                    estimated_rent = estimate_rent_alternative(df,label_encoders,area=input_property['builtup_area'],locality=input_property['locality'],society=input_property['society'],furnishing=input_property['furnishing'])
                    st.session_state.last_selected_society = selected_society
                    st.session_state.last_selected_bhk = selected_bhk

                # Display results in columns
                col1, col2, col3 = st.columns(3)
    
                with col1:
                    st.metric("Model A (Raw Rent)", f"₹{results['model_a_raw_prediction']:,.0f}/month")
    
                with col2:
                    st.metric("Model B (Log Rent)", f"₹{results['model_b_log_prediction']:,.0f}/month")
    
                with col3:
                    st.metric("Rent/sqft Estimate", f"₹{estimated_rent:,.0f}/month")
    
                
                # Display explanation
                with st.expander("Model Explanation"):
                    st.write("""
                    ### Prediction Models
                    
                    **Model A (Raw Rent)**: Trained on actual rent values. Performs well for properties in the middle price range.
                    
                    **Model B (Log Rent)**: Trained on log-transformed rent values, which helps to handle the skewed distribution of rent prices. Often performs better for high-end or low-end properties.
                                
                    **Rent/sqft Estimate**: Calculates the average rent per square foot for the selected society (or locality if society data is insufficient) and multiplies it by the area. Adjusted based on furnishing type.
    
                    """)
                    
                    # Display model metrics
                    st.write("### Model Performance Metrics")
                    metrics_df = pd.DataFrame({
                        'Model': ['Model A (Raw)', 'Model B (Log-transformed)'],
                        'MAE': [f"₹{models['mae_a']:,.0f}", f"₹{models['mae_b']:,.0f}"],
                        'RMSE': [f"₹{models['rmse_a']:,.0f}", f"₹{models['rmse_b']:,.0f}"]
                    })
                    st.table(metrics_df)
                with st.expander("Explain Prediction (SHAP Analysis)"):
                    st.write("This visualization shows how each feature contributes to the predicted rent value:")
                    
                    # Create input DataFrame for SHAP analysis
                    features_with_log = [f if f != 'builtup_area' else 'log_builtup_area' for f in models['features']]
                    input_df = pd.DataFrame([input_property]).copy()
                    
                    # Feature engineering
                    input_df['floor'] = input_df['floor'].replace(0, 1)
                    input_df['total_floors'] = input_df['total_floors'].replace(0, 1)
                    
                    if 'floor' in input_df.columns and 'total_floors' in input_df.columns:
                        input_df['floor_to_total_floors'] = input_df['floor'] / input_df['total_floors']
                    
                    # Add log transformation for builtup_area
                    input_df['log_builtup_area'] = np.log1p(input_df['builtup_area'])
                    
                    # Encode categorical variables
                    for col in ['furnishing', 'locality', 'society']:
                        try:
                            if col in input_df.columns and col in label_encoders:
                                input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
                        except ValueError as e:
                            st.error(f"Value '{input_df[col][0]}' not found in training data for column {col}")
                    
                    # Select model features
                    input_features = input_df[features_with_log]
                    
                    try:
                        # Generate SHAP waterfall plot
                        fig = generate_shap_waterfall(
                            models['model_a'],
                            features_with_log,
                            input_features,
                            features_with_log,
                            label_encoders
                        )
                        st.pyplot(fig)
                        
                        st.write("### How to interpret this chart:")
                        st.write("""
                        - The base value is the average prediction across all properties
                        - Red arrows show features pushing the rent higher
                        - Blue arrows show features pushing the rent lower
                        - The final prediction is shown at the bottom
                        """)
                    except Exception as e:
                        st.error(f"Could not generate SHAP explanation: {e}")

        
        # Comparable Properties Tab
        with tabs[1]:
            st.header("Comparable Properties")
            with st.form(key="comparable_search_form"):
    
                # Input for comparable search
                col1, col2 = st.columns(2)
                
                with col1:
                    default_society = st.session_state.last_selected_society if st.session_state.last_selected_society else unique_societies[0]
                    default_bhk = st.session_state.last_selected_bhk if st.session_state.last_selected_bhk else 3.0
                    comp_society = st.selectbox("Select Society", unique_societies, index=unique_societies.index(default_society) if default_society in unique_societies else 0, key="comp_society")
                    comp_bhk = st.number_input("Bedrooms", min_value=1.0, max_value=7.0, value=default_bhk, step=0.5, key="comp_bhk")
                
                with col2:
                    radius_km = st.slider("Search Radius (km)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
                    # Add an option to show a map of nearby properties
                    show_map = st.checkbox("Show properties on map", value=False)

                # Submit button
                find_submitted = st.form_submit_button("Find Comparable Properties")

                # Find button
                if find_submitted:
                    # Get coordinates from a sample property
                    try:
                        society_encoded = label_encoders['society'].transform([comp_society])[0]
                        same_society_df = df[df['society'] == society_encoded]
                        
                        if not same_society_df.empty:
                            sample_row = same_society_df.iloc[0]
                            lat, lon = sample_row['latitude'], sample_row['longitude']
                            
                            # Find comparables
                            same_society, same_bhk, nearby = find_comparables(
                                df, label_encoders, comp_society, comp_bhk, lat, lon, radius_km
                            )
                            
                            # Display counts
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Same Society", f"{same_society.shape[0]} properties")
                            with col2:
                                st.metric(f"Same BHK ({comp_bhk})", f"{same_bhk.shape[0]} properties")
                            with col3:
                                st.metric(f"Within {radius_km}km", f"{nearby.shape[0]} properties")
                            
                            # Show properties on tabs
                            comp_tabs = st.tabs(["Same Society", "Same BHK", "Nearby Properties"])
                            
                            # Function to display property dataframe
                            def display_properties(properties_df, label_encoders):
                                if not properties_df.empty:
                                    # Create a copy to avoid modifying the original
                                    display_df = properties_df.copy()
                                    
                                    # Decode categorical columns
                                    for col in ['society', 'locality', 'furnishing']:
                                        if col in display_df.columns:
                                            display_df[col] = display_df[col].apply(
                                                lambda x: label_encoders[col].inverse_transform([x])[0]
                                            )
                                    
                                    # Select and rename columns for display
                                    cols_to_display = ['society', 'locality', 'bedrooms', 'bathrooms', 
                                                        'builtup_area', 'furnishing', 'rent', 'floor', 
                                                        'total_floors']
                                    
                                    if 'distance_km' in display_df.columns:
                                        cols_to_display.append('distance_km')
                                    
                                    display_df = display_df[[c for c in cols_to_display if c in display_df.columns]]
                                    
                                    # Format rent as currency
                                    if 'rent' in display_df.columns:
                                        display_df['rent'] = display_df['rent'].apply(lambda x: f"₹{x:,.0f}")
                                    
                                    # Calculate and add rent per sqft
                                    if 'rent' in properties_df.columns and 'builtup_area' in properties_df.columns:
                                        display_df['rent_per_sqft'] = (
                                            pd.to_numeric(properties_df['rent']) / 
                                            properties_df['builtup_area']
                                        ).apply(lambda x: f"₹{x:.1f}")
                                    
                                    # Format distance if present
                                    if 'distance_km' in display_df.columns:
                                        display_df['distance_km'] = display_df['distance_km'].apply(lambda x: f"{x:.2f} km")
                                    
                                    # Display the dataframe
                                    st.dataframe(display_df)
                                else:
                                    st.info("No properties found.")
                            
                            # Display properties in each tab
                            with comp_tabs[0]:
                                display_properties(same_society, label_encoders)
                            
                            with comp_tabs[1]:
                                display_properties(same_bhk, label_encoders)
                            
                            with comp_tabs[2]:
                                display_properties(nearby, label_encoders)
                            if show_map and not nearby.empty:
                                st.subheader("Map of Nearby Properties")
                                # Create map using st.map if latitude and longitude are available
                                map_data = nearby[['latitude', 'longitude']].copy()
                                # Add the current property to highlight it
                                map_data = pd.concat([
                                    map_data, 
                                    pd.DataFrame({'latitude': [lat], 'longitude': [lon]})
                                ])
                                st.map(map_data)                        
                        else:
                            st.error(f"No properties found in {comp_society}.")
                    except Exception as e:
                        st.error(f"Error finding comparable properties: {e}")
        
        # Data Exploration Tab
        with tabs[2]:
            st.header("Data Exploration")
            
            # Show basic statistics
            st.subheader("Dataset Summary")
            st.write(f"Total properties: {len(df)}")
            
            # Distribution of properties by bedroom count
            if 'bedrooms' in df.columns:
                st.subheader("Properties by Bedroom Count")
                bedroom_counts = df['bedrooms'].value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(10, 6))
                bedroom_counts.plot(kind='bar', ax=ax)
                ax.set_xlabel('Number of Bedrooms')
                ax.set_ylabel('Count')
                ax.set_title('Distribution of Properties by Bedroom Count')
                st.pyplot(fig)
            
            # Price distribution
            if 'rent' in df.columns:
                st.subheader("Rent Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df['rent'], bins=20, kde=True, ax=ax)
                ax.set_xlabel('Monthly Rent (₹)')
                ax.set_ylabel('Count')
                ax.set_title('Distribution of Monthly Rent')
                st.pyplot(fig)
            
            # Correlation heatmap
            st.subheader("Correlation Heatmap")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            corr_matrix = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Between Numeric Features')
            st.pyplot(fig)
        # Landlord Report Tab
        with tabs[3]:
            st.header("🏠 Landlord Report")
            st.markdown("""
            Generate a comprehensive market analysis report for your rental property. 
            This report includes rent estimates, market position analysis, comparable properties, 
            and actionable recommendations.
            """)
            
            # Input columns layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Get unique societies and localities
                unique_societies = list(label_encoders['society'].classes_) if 'society' in label_encoders else []
                unique_localities = list(label_encoders['locality'].classes_) if 'locality' in label_encoders else []
                
                # Property inputs
                selected_locality = st.selectbox("Locality", unique_localities, key="lr_locality")
                selected_society = st.selectbox("Society", unique_societies, key="lr_society")
                
            with col2:
                selected_bhk = st.number_input("Bedrooms", min_value=1.0, max_value=7.0, value=3.0, step=0.5, key="lr_bhk")
                selected_bathrooms = st.number_input("Bathrooms", min_value=1, max_value=7, value=2, key="lr_bath")
                selected_area = st.number_input("Built-up Area (sqft)", min_value=100, max_value=10000, value=1500, key="lr_area")
                
            with col3:
                furnishing_options = list(label_encoders['furnishing'].classes_) if 'furnishing' in label_encoders else []
                selected_furnishing = st.selectbox("Furnishing", furnishing_options, key="lr_furn")
                selected_floor = st.number_input("Floor", min_value=1, max_value=50, value=5, key="lr_floor")
                selected_total_floors = st.number_input("Total Floors", min_value=1, max_value=50, value=10, key="lr_total_floors")
                
            # Optional fields
            with st.expander("Additional Fields"):
                building_age = st.slider("Building Age (years)", min_value=0, max_value=30, value=5, key="lr_age")
                property_id = st.text_input("Property ID (optional)", value="PROP001", key="lr_id")
                
            # Create input property dictionary
            input_property = {
                'property_id': property_id,
                'bedrooms': selected_bhk,
                'builtup_area': selected_area,
                'bathrooms': selected_bathrooms,
                'furnishing': selected_furnishing,
                'locality': selected_locality,
                'society': selected_society,
                'floor': selected_floor,
                'total_floors': selected_total_floors,
                'building_age': building_age,
                'floor_to_total_floors': selected_floor / selected_total_floors
            }
            
            # Report options
            option_col1, option_col2 = st.columns(2)
            with option_col1:
                generate_plots = st.checkbox("Include visualizations", value=True, key="lr_viz")
            with option_col2:
                use_model_b = st.checkbox("Use log-transformed model", value=True, 
                                        help="Log-transformed model often performs better for high-end properties")
            
            # Generate Report button
            if st.button("Generate Landlord Report", type="primary", key="generate_report_button"):
                # Calculate rent predictions
                with st.spinner("Calculating rent estimates..."):
                    # ML-based Prediction
                    results = predict_rent_dual(input_property, models, label_encoders)
                    
                    # Area-based alternative estimate
                    estimated_rent = estimate_rent_alternative(df, label_encoders, 
                                                             area=input_property['builtup_area'],
                                                             locality=input_property['locality'],
                                                             society=input_property['society'],
                                                             furnishing=input_property['furnishing'])
                    
                    # Calculate total rent for the report
                    input_property['total_rent'] = (results['model_a_raw_prediction'] + 
                                                  results['model_b_log_prediction'] + 
                                                  estimated_rent) / 3
                    
                    # Calculate rent per sqft
                    input_property['rent_per_sqft'] = input_property['total_rent'] / input_property['builtup_area']
                    
                    # Display the rent estimates
                    st.subheader("Rent Estimates")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Model A", f"₹{results['model_a_raw_prediction']:,.0f}/month")
                    col2.metric("Model B", f"₹{results['model_b_log_prediction']:,.0f}/month")
                    col3.metric("Area-based", f"₹{estimated_rent:,.0f}/month")
                    
                    # Calculate combined estimate
                    combined_estimate = (results['model_a_raw_prediction'] + 
                                        results['model_b_log_prediction'] + 
                                        estimated_rent) / 3
                    
                    # Show combined estimate
                    st.metric("Combined Estimate", f"₹{combined_estimate:,.0f}/month")
                    
                    # Package rent estimates for the report
                    rent_estimates = {
                        'model_a': results['model_a_raw_prediction'],
                        'model_b': results['model_b_log_prediction'],
                        'sqft_method': estimated_rent,
                        'combined_estimate': combined_estimate
                    }
                
                # Generate the full report
                with st.spinner("Generating comprehensive landlord report..."):
                    # Generate the landlord report
                    landlord_report = generate_landlord_report(
                        input_property,  # Use the input_property directly
                        df,
                        ml_model=models['model_b'] if use_model_b else models['model_a'],
                        feature_names=models['features'],
                        label_encoders=label_encoders,
                        generate_plots=generate_plots,
                        rent_estimates=rent_estimates
                    )
                    
                    # Display report sections
                    
                    # 1. Property details section
                    st.subheader("📍 Property Details")
                    details = landlord_report['property_details']
                    
                    # Create a formatted table for property details
                    property_data = [
                        ["Locality", details['location']['locality']],
                        ["Society", details['location']['society']],
                        [f"Configuration", f"{details['physical']['bedrooms']} BHK, {details['physical']['bathrooms']} Bathrooms"],
                        ["Built-up Area", f"{details['physical']['builtup_area']:,} sq.ft."],
                        ["Floor", f"{details['building']['floor']} out of {details['building']['total_floors']}"],
                        ["Furnishing", details['condition']['furnishing']],
                        ["Monthly Rent", f"₹{details['pricing']['total_rent']:,.0f}"],
                        ["Rent per sq.ft.", f"₹{details['pricing']['rent_per_sqft']:.2f}"]
                    ]
                    
                    # Display as a DataFrame for better formatting
                    st.table(pd.DataFrame(property_data, columns=["Parameter", "Value"]))
                    
                    # 2. Market position section with percentile rankings
                    st.subheader("📊 Market Position Analysis")
                    
                    market_position = landlord_report['market_position']
                    position_category = market_position['position_category']
                    primary_group = market_position['primary_comparison_group']
                    
                    # Format the primary comparison group name
                    primary_group_display = primary_group.replace('_', ' ').title()
                    
                    st.write(f"Your property is positioned in the **{position_category}** segment, based on comparison with {primary_group_display} properties.")
                    
                    # Display percentile rankings
                    st.write("#### Percentile Rankings")
                    
                    percentile_data = []
                    percentile_data.append(["Comparison Group", "Rent Percentile", "Rent/sq-ft Percentile"])
                    
                    for group_name, values in market_position['percentile_ranks'].items():
                        if group_name != 'bedroom_type':  # Skip this for cleaner report
                            group_display = group_name.replace('_', ' ').title()
                            rent_percentile = values.get('rent', 0)
                            rent_sqft_percentile = values.get('rent_sqft', 0)
                            
                            percentile_data.append([
                                group_display,
                                f"{rent_percentile:.1f}th",
                                f"{rent_sqft_percentile:.1f}th"
                            ])
                    
                    # Display as a DataFrame for better formatting
                    st.table(pd.DataFrame(percentile_data[1:], columns=percentile_data[0]))
                    
                    # Your market position progress bar
                    st.write("#### Your Market Position")
                    
                    # Get the appropriate percentile for the progress bar from same comparison group as in PDF
                    bar_percentile = 0
                    if 'same_society_same_bhk' in market_position['percentile_ranks']:
                        bar_percentile = market_position['percentile_ranks']['same_society_same_bhk'].get('rent_sqft', 0)
                    elif 'overall_market' in market_position['percentile_ranks']:
                        bar_percentile = market_position['percentile_ranks']['overall_market'].get('rent_sqft', 0)
                    
                    # Create a progress bar
                    st.progress(bar_percentile / 100)
                    
                    # Add labels for clarity
                    col1, col2, col3 = st.columns([1, 1, 1])
                    col1.write("Below Market")
                    col2.write("At Market")
                    col3.write("Above Market")
                    
                    st.write(f"You are at the **{bar_percentile:.1f}th percentile** of the market")
                    
                    # Premium/Discount Analysis
                    st.write("#### Premium/Discount Analysis")
                    
                    premium_data = []
                    premium_data.append(["Comparison Group", "Premium/Discount"])
                    
                    # Helper function to format premium/discount values
                    def format_premium(value):
                        status = "PREMIUM" if value > 0 else "DISCOUNT"
                        return f"{abs(value):.1f}% {status}"
                    
                    premium_discount = market_position['premium_discount']
                    
                    # Use the same groups as in the PDF report
                    society_premium = premium_discount.get('same_society_same_bhk_avg', 
                                                        premium_discount.get('society_avg', 0))
                    locality_premium = premium_discount.get('same_locality_same_bhk_avg', 
                                                         premium_discount.get('locality_avg', 0))
                    overall_premium = premium_discount.get('overall_market_avg', 
                                                        premium_discount.get('locality_avg', 0))
                    comparables_premium = premium_discount.get('comparables_avg', overall_premium)
                    
                    premium_data.extend([
                        ["Same Society & BHK", format_premium(society_premium)],
                        ["Same Locality & BHK", format_premium(locality_premium)],
                        ["Overall Market", format_premium(overall_premium)],
                        ["Primary Comparables", format_premium(comparables_premium)]
                    ])
                    
                    # Display as a DataFrame for better formatting
                    st.table(pd.DataFrame(premium_data[1:], columns=premium_data[0]))
                    
                    # 3. Comparable properties analysis
                    st.subheader("🏘️ Comparable Properties Analysis")
                    
                    st.write("The following analysis shows how your property compares to different segments of the market:")
                    
                    # Create tabs for different tiers of comparables
                    comp_tabs = st.tabs([
                        "Same Society & BHK", 
                        "Same Locality & BHK",
                        "Same Society Different BHK", 
                        "Same Locality Similar BHK"
                    ])
                    
                    # Define a function to display comparable tier data
                    def display_comparable_tier(tier_data, tier_name):
                        if not tier_data.get('available', False) or tier_data.get('count', 0) < 3:
                            st.info(f"Insufficient {tier_name} properties for analysis.")
                            return
                        
                        # Show basic stats
                        cols = st.columns(4)
                        cols[0].metric("Properties", tier_data['count'])
                        cols[1].metric("Avg. Rent", f"₹{tier_data['avg_rent']:,.0f}")
                        cols[2].metric("Min. Rent", f"₹{tier_data['min_rent']:,.0f}")
                        cols[3].metric("Max. Rent", f"₹{tier_data['max_rent']:,.0f}")
                        
                        # Show premium/discount 
                        premium = tier_data.get('premium_discount', 0)
                        premium_text = f"{abs(premium):.1f}% {'PREMIUM' if premium > 0 else 'DISCOUNT'}"
                        
                        # Create a progress bar to visualize position
                        st.write(f"**Your property vs. {tier_name}:** {premium_text}")
                        
                        # Calculate position for progress bar (0-1 scale)
                        position = 0.5 + (premium / 200)  # Center is market rate, right is premium, left is discount
                        position = max(0, min(1, position))  # Clamp between 0 and 1
                        
                        # Display progress bar
                        st.progress(position)
                        
                        # Add labels
                        cols = st.columns([1, 2, 1])
                        cols[0].write("Below market")
                        cols[1].write("At market")
                        cols[2].write("Above market")
                        
                        # Add rent per square foot analysis if available
                        if 'avg_rent_psf' in tier_data:
                            st.write("**Rent per Square Foot Analysis**")
                            cols = st.columns(3)
                            cols[0].metric("Avg. ₹/sqft", f"₹{tier_data['avg_rent_psf']:.2f}")
                            
                            # Show premium/discount for rent per sqft
                            premium_psf = tier_data.get('premium_discount_psf', 0)
                            premium_psf_text = f"{abs(premium_psf):.1f}% {'PREMIUM' if premium_psf > 0 else 'DISCOUNT'}"
                            st.write(f"**Your property's ₹/sqft vs. {tier_name}:** {premium_psf_text}")
                    
                    # Display each tier in its tab
                    with comp_tabs[0]:
                        display_comparable_tier(
                            landlord_report['comparables']['tiered_analysis'].get('same_society_same_bhk', {}), 
                            "Same Society & BHK"
                        )
                    
                    with comp_tabs[1]:
                        display_comparable_tier(
                            landlord_report['comparables']['tiered_analysis'].get('same_locality_same_bhk', {}), 
                            "Same Locality & BHK"
                        )
                    
                    with comp_tabs[2]:
                        display_comparable_tier(
                            landlord_report['comparables']['tiered_analysis'].get('same_society_diff_bhk', {}), 
                            "Same Society Different BHK"
                        )
                    
                    with comp_tabs[3]:
                        display_comparable_tier(
                            landlord_report['comparables']['tiered_analysis'].get('same_locality_similar_bhk', {}), 
                            "Same Locality Similar BHK"
                        )
                    
                    # 4. Market visualizations if enabled
                    if generate_plots:
                        st.subheader("📈 Market Visualizations")
                        
                        # Create tabs for different visualizations
                        viz_tabs = st.tabs([
                            "Position Chart", 
                            "Feature Comparison", 
                            "Feature Impact", 
                            "Market Distribution"
                        ])
                        
                        # Position chart
                        with viz_tabs[0]:
                            st.write("### Your Property vs. Comparable Properties")
                            st.write("This chart shows how your property's rent (red line) compares to similar properties in the same market segment.")
                            
                            # Get or generate the visualization
                            position_chart = None
                            if 'position_chart' in landlord_report.get('visualizations', {}):
                                position_chart = landlord_report['visualizations']['position_chart']
                            elif 'position_chart' in landlord_report.get('visualization_generators', {}):
                                try:
                                    position_chart = landlord_report['visualization_generators']['position_chart']()
                                except Exception as e:
                                    st.error(f"Error generating position chart: {str(e)}")
                            
                            if position_chart:
                                st.pyplot(position_chart)
                            else:
                                st.info("Position chart not available for this property.")
                        
                        # Feature radar chart
                        with viz_tabs[1]:
                            st.write("### Property Feature Comparison")
                            st.write("This radar chart compares your property's key features to comparable properties. Areas where your property's line (red) extends further than others indicate competitive strengths.")
                            
                            # Get or generate the visualization
                            feature_radar = None
                            if 'feature_radar' in landlord_report.get('visualizations', {}):
                                feature_radar = landlord_report['visualizations']['feature_radar']
                            elif 'feature_radar' in landlord_report.get('visualization_generators', {}):
                                try:
                                    feature_radar = landlord_report['visualization_generators']['feature_radar']()
                                except Exception as e:
                                    st.error(f"Error generating feature radar: {str(e)}")
                            
                            if feature_radar:
                                st.pyplot(feature_radar)
                            else:
                                st.info("Feature comparison chart not available for this property.")
                        
                        # Feature impact chart
                        with viz_tabs[2]:
                            st.write("### Factors Affecting Rent Value")
                            st.write("This chart shows how different features contribute to your property's rent value. Green bars indicate positive contributions, red bars indicate negative impacts.")
                            
                            # Get or generate the visualization
                            feature_impact = None
                            if 'feature_impact' in landlord_report.get('visualizations', {}):
                                feature_impact = landlord_report['visualizations']['feature_impact']
                            elif 'feature_impact' in landlord_report.get('visualization_generators', {}):
                                try:
                                    feature_impact = landlord_report['visualization_generators']['feature_impact']()
                                except Exception as e:
                                    st.error(f"Error generating feature impact: {str(e)}")
                            
                            if feature_impact:
                                st.pyplot(feature_impact)
                            else:
                                st.info("Feature impact chart not available for this property.")
                        
                        # Market distribution chart
                        with viz_tabs[3]:
                            st.write("### Market Rent Distribution")
                            
                            # Add percentile info
                            percentile = landlord_report['market_position']['percentile_ranks'].get('overall_market', {}).get('rent', 0)
                            st.write(f"This chart shows your property's position in the overall market rent distribution. Your property is at the **{percentile:.0f}th percentile** of the market.")
                            
                            # Get or generate the visualization
                            rent_distribution = None
                            if 'rent_distribution' in landlord_report.get('visualizations', {}):
                                rent_distribution = landlord_report['visualizations']['rent_distribution']
                            elif 'rent_distribution' in landlord_report.get('visualization_generators', {}):
                                try:
                                    rent_distribution = landlord_report['visualization_generators']['rent_distribution']()
                                except Exception as e:
                                    st.error(f"Error generating rent distribution: {str(e)}")
                            
                            if rent_distribution:
                                st.pyplot(rent_distribution)
                            else:
                                st.info("Rent distribution chart not available for this property.")
                    
                    # 5. Summary and recommendations
                    st.subheader("📑 Summary & Recommendations")
                    
                    # Basic summary text
                    property_rent = landlord_report['property_details']['pricing']['total_rent']
                    position_category = landlord_report['market_position']['position_category']
                    
                    st.write(f"Your property is currently positioned in the **{position_category}** segment of the market with a monthly rent of ₹{property_rent:,}.")
                    
                    # Get premium/discount from most relevant comparable group
                    primary_premium = landlord_report['market_position']['premium_discount'].get('comparables_avg', 0)
                    
                    # Market position insights based on category
                    st.write("#### Market Position Insights")
                    
                    if position_category in ["Premium", "Above Market"]:
                        if primary_premium > 15:
                            st.write("Your property commands a significant premium over comparable properties. This premium positioning should be supported by maintaining excellent property condition and amenities to justify the higher rent.")
                        else:
                            st.write("Your property is positioned above the market average, indicating strong features or amenities that tenants value. This is a good position that balances revenue optimization with occupancy.")
                    elif position_category == "At Market":
                        st.write("Your property is priced in line with the market, which should help balance rental income with minimal vacancy periods. This is typically an optimal position for steady income.")
                    elif position_category in ["Below Market", "Significantly Below Market"]:
                        if primary_premium < -15:
                            st.write("Your property is priced significantly below comparable properties. There may be opportunity to increase rent gradually, especially if you make improvements to the property or when renewing leases.")
                        else:
                            st.write("Your property is priced somewhat below the market, which can help minimize vacancy but may mean you're leaving potential rental income on the table.")
                    else:
                        st.write("Insufficient data to provide specific market position insights.")
                    
                    # Recommendations
                    st.write("#### Recommendations")
                    
                    # Create recommendations based on market position
                    recommendations = []
                    
                    if position_category in ["Premium", "Above Market"]:
                        recommendations = [
                            "Maintain high standards of property maintenance and amenities to justify the premium rent",
                            "Consider investing in property upgrades that will help maintain your competitive advantage"
                        ]
                        if primary_premium > 20:
                            recommendations.append("Monitor vacancy periods closely as very high premiums can lead to longer vacancies")
                    elif position_category == "At Market":
                        recommendations = [
                            "Consider modest rent increases in line with market growth to maintain position",
                            "Focus on tenant retention as your pricing is competitive",
                            "Monitor market trends to ensure your property stays aligned with comparable properties"
                        ]
                    else:  # Below Market
                        recommendations = [
                            "Consider a moderate rent increase with your next lease renewal",
                            "Evaluate if there are property improvements that could justify higher rent",
                            "Review your tenant acquisition strategy as below-market rent should allow for selectivity"
                        ]
                    
                    # Display recommendations as bullets
                    for rec in recommendations:
                        st.write(f"• {rec}")
                    
                    # Generate PDF button
                    st.write("#### Download Full Report")
                    st.write("Click below to generate and download a detailed PDF report.")
                    
                    if st.button("Generate PDF Report", key="generate_pdf_button"):
                        with st.spinner("Creating PDF report..."):
                            try:
                                # Ensure reports directory exists
                                import os
                                os.makedirs("reports", exist_ok=True)
                                
                                # Generate the PDF
                                pdf_path = create_landlord_pdf_report(
                                    landlord_report, 
                                    label_encoders=label_encoders,
                                    output_dir="reports"
                                )
                                
                                # Read the generated PDF for download
                                with open(pdf_path, "rb") as pdf_file:
                                    pdf_bytes = pdf_file.read()
                                
                                # Display download button
                                st.success("✅ PDF report generated successfully!")
                                st.download_button(
                                    label="Download PDF Report",
                                    data=pdf_bytes,
                                    file_name=f"Landlord_Report_{property_id}.pdf",
                                    mime="application/pdf"
                                )
                            except Exception as e:
                                st.error(f"Error creating PDF report: {str(e)}")
                                st.info("Please check if the reports directory is writable.")
                    
                    # Add disclaimer
                    st.caption("Note: This analysis is based on current market data and is intended for informational purposes only. Actual market conditions may vary.")
else:
    st.error("Error processing the data. Please check your CSV file.")
    
    # Show expected data format
    with st.expander("Expected Data Format"):
        st.markdown("""
        Your CSV file should contain the following columns:
        
        - **locality** or **property_localityname**: Area or neighborhood
        - **society** or **detail_projectname**: Name of housing society/apartment
        - **property_type** or **detail_propertytype**: Type of property (e.g., 'multistorey apartment')
        - **builtup_area** or **detail_coveredarea**: Built-up area in square feet
        - **bedrooms** or **detail_bedrooms**: Number of bedrooms (BHK)
        - **bathrooms** or **detail_bathrooms**: Number of bathrooms
        - **furnishing** or **detail_furnished**: Furnishing status (e.g., unfurnished, semi-furnished, fully furnished)
        - **floor** or **detail_floornumber**: Floor number
        - **total_floors** or **detail_totalfloornumber**: Total number of floors
        - **rent** or **detail_exactsalerentprice**: Monthly rent amount
        - **latitude** or **detail_latitude**: Geographic coordinates (latitude)
        - **longitude** or **detail_longitude**: Geographic coordinates (longitude)
        - **detail_additionalrooms**: Additional rooms info (optional)
        - **building_age** or **detail_ageofcons**: Age of building (optional)
        """)
