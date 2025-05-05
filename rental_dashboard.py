import streamlit as st
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from geopy.distance import geodesic
import os
import io
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Real Estate Rental Analysis Dashboard",
    page_icon="🏠",
    layout="wide"
)

# App title and description
st.title("🏠 Real Estate Rental Analysis Dashboard")
st.markdown("""
This app analyzes rental property data to predict rents and find comparable properties.
Upload your CSV file with property data to get started.
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
    st.write(f"Looking for file at: {file_path}")
    st.write(f"File exists: {os.path.exists(file_path)}")

    
    try:
        df_raw = pd.read_csv(file_path)
        st.write(f"Successfully loaded file with {len(df_raw)} rows")


        # Read the uploaded file
        df_raw = pd.read_csv(uploaded_file)
        
        # Rename and normalize column names
        df_raw.columns = df_raw.columns.str.strip().str.lower()
        
        # Define column mapping
        rename_map = {
            'property_localityname': 'locality',
            'detail_propertytype': 'property_type',
            'detail_coveredarea': 'builtup_area',
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
        
        # Extract amenity columns
        amenity_cols = [col for col in df.columns if col.startswith('detail_amenitymap_') or col.startswith('detail_amenityexternalmap_')]
        amenity_df = df[amenity_cols].copy()
        amenity_df.columns = [col.replace('detail_amenitymap_', '').replace('detail_amenityexternalmap_', '') for col in amenity_cols]
        amenity_df = amenity_df.fillna(0).applymap(lambda x: 1 if str(x).strip().lower() in ['true', '1', 'yes', 'present'] else 0)

        # Concatenate back to main df
        df = pd.concat([df, amenity_df], axis=1)
        # Filter for necessary columns
        required_cols = [
            'locality', 'society', 'property_type', 'builtup_area', 'bedrooms', 'bathrooms', 'furnishing',
            'floor', 'total_floors', 'rent', 'latitude', 'longitude', 'detail_additionalrooms', 'building_age'
        ]
        existing_cols = [col for col in required_cols if col in df.columns]
        df = df[existing_cols].copy() if existing_cols else df
        
        # Filter for multi-storey apartments if column exists
        if 'property_type' in df.columns:
            df = df[df['property_type'].str.lower().str.strip() == 'multistorey apartment']
        
        # Drop rows with missing values in critical fields
        critical_fields = ['rent', 'builtup_area', 'bedrooms', 'locality', 'society']
        critical_fields = [col for col in critical_fields if col in df.columns]
        if critical_fields:
            df = df.dropna(subset=critical_fields)
        
        # Convert numeric columns
        numeric_cols = {
            'rent': 0,
            'builtup_area': 0,
            'bedrooms': 0,
            'bathrooms': 0,
            'floor': 1,
            'total_floors': 1
        }
        for col, default in numeric_cols.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)
        
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
        
        for col in ['rent', 'builtup_area', 'total_floors']:
            if col in df.columns:
                df = remove_outliers(df, col)
        
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
        
        return df, label_encoders
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


# Function to train prediction models
def train_models(df):
    if df is None:
        return None
    
    features = ['bedrooms', 'builtup_area', 'bathrooms', 'furnishing', 'locality', 
               'society', 'floor', 'total_floors', 'floor_to_total_floors', 'building_age']
    
    # Make sure all features exist
    features = [f for f in features if f in df.columns]
    
    if not features or 'total_rent' not in df.columns:
        st.error("Missing required columns for model training")
        return None
    
    try:
        X = df[features]
        y_a = df['total_rent']
        y_b = df['log_total_rent']
        
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
        input_features = input_df[models['features']]
        
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
        tabs = st.tabs(["Rent Prediction", "Comparable Properties", "Data Exploration"])
        
        # Rent Prediction Tab
        with tabs[0]:
            st.header("Rent Prediction")
            
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
            if st.button("Predict Rent", key="predict_button"):
                with st.spinner("Predicting..."):
                    # ML-based Prediction
                    results = predict_rent_dual(input_property, models, label_encoders)
                    estimated_rent = estimate_rent_alternative(df,label_encoders,area=input_property['builtup_area'],locality=input_property['locality'],society=input_property['society'],furnishing=input_property['furnishing'])

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
        
        # Comparable Properties Tab
        with tabs[1]:
            st.header("Comparable Properties")
            
            # Input for comparable search
            col1, col2 = st.columns(2)
            
            with col1:
                comp_society = st.selectbox("Select Society", unique_societies, key="comp_society")
                comp_bhk = st.number_input("Bedrooms", min_value=1.0, max_value=7.0, value=3.0, step=0.5, key="comp_bhk")
            
            with col2:
                radius_km = st.slider("Search Radius (km)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
            
            # Find button
            if st.button("Find Comparable Properties"):
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
