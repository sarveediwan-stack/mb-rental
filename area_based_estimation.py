"""
Area-Based Rent Estimation Module

This module provides improved area-based rent estimation using hierarchical 
comparable property matching with confidence scoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy.stats import percentileofscore

def estimate_rent_improved_area_based(
    property_data: Dict,
    full_dataset: pd.DataFrame,
    label_encoders: Optional[Dict] = None
) -> Dict:
    """
    Improved area-based rent estimation using hierarchical comparable property matching
    
    Parameters:
    property_data: Dict with property features (society, locality, bedrooms, area, furnishing, floor, total_floors)
    full_dataset: DataFrame with all properties
    label_encoders: Dictionary of label encoders for categorical variables
    
    Returns:
    Dict with estimate, confidence, range, and explanation
    """
    
    # Extract property details
    target_society = property_data.get('society')
    target_locality = property_data.get('locality')
    target_bedrooms = property_data.get('bedrooms')
    target_area = property_data.get('builtup_area')
    target_furnishing = property_data.get('furnishing')
    target_floor = property_data.get('floor', 1)
    target_total_floors = property_data.get('total_floors', 1)
    
    # Encode categorical values if needed
    if label_encoders:
        try:
            if isinstance(target_society, str) and 'society' in label_encoders:
                target_society_encoded = label_encoders['society'].transform([target_society])[0]
            else:
                target_society_encoded = target_society
                
            if isinstance(target_locality, str) and 'locality' in label_encoders:
                target_locality_encoded = label_encoders['locality'].transform([target_locality])[0]
            else:
                target_locality_encoded = target_locality
                
            if isinstance(target_furnishing, str) and 'furnishing' in label_encoders:
                target_furnishing_encoded = label_encoders['furnishing'].transform([target_furnishing])[0]
            else:
                target_furnishing_encoded = target_furnishing
        except:
            # If encoding fails, use original values
            target_society_encoded = target_society
            target_locality_encoded = target_locality
            target_furnishing_encoded = target_furnishing
    else:
        target_society_encoded = target_society
        target_locality_encoded = target_locality
        target_furnishing_encoded = target_furnishing
    
    # Step 1: Define hierarchical tiers
    tiers = [
        {
            'name': 'Same Society, Same BHK',
            'filter_func': lambda df: df[
                (df['society'] == target_society_encoded) & 
                (df['bedrooms'] == target_bedrooms)
            ],
            'confidence_base': 95
        },
        {
            'name': 'Same Society, Different BHK',
            'filter_func': lambda df: df[
                (df['society'] == target_society_encoded) & 
                (df['bedrooms'] != target_bedrooms)
            ],
            'confidence_base': 85
        },
        {
            'name': 'Same Locality, Same BHK',
            'filter_func': lambda df: df[
                (df['locality'] == target_locality_encoded) & 
                (df['bedrooms'] == target_bedrooms)
            ],
            'confidence_base': 75
        },
        {
            'name': 'Same Locality, Different BHK',
            'filter_func': lambda df: df[
                (df['locality'] == target_locality_encoded) & 
                (df['bedrooms'] != target_bedrooms)
            ],
            'confidence_base': 60
        }
    ]
    
    # Step 2: Find the best tier with sufficient data
    selected_tier = None
    tier_data = None
    
    for tier in tiers:
        try:
            tier_properties = tier['filter_func'](full_dataset)
            if len(tier_properties) >= 5:
                selected_tier = tier
                tier_data = tier_properties
                break
        except Exception as e:
            print(f"Error filtering tier {tier['name']}: {e}")
            continue
    
    # Step 3: Fallback to city-wide average if no tier has enough data
    if selected_tier is None:
        return area_based_calculate_city_average_estimate(full_dataset, property_data)
    
    # Step 4: Outlier detection and filtering
    tier_data = area_based_remove_outliers(tier_data)
    
    if len(tier_data) < 3:  # If outlier removal left us with too few properties
        return area_based_calculate_city_average_estimate(full_dataset, property_data)
    
    # Step 5: Calculate baseline rent per sqft
    baseline_result = area_based_calculate_baseline_rent_per_sqft(tier_data, label_encoders)
    baseline_rent_per_sqft = baseline_result['baseline_rent_per_sqft']
    furnishing_used = baseline_result['furnishing_used']
    print(furnishing_used)
    properties_count = baseline_result['properties_count']
    
    # Step 6: Apply adjustments
    
    # A. Area adjustment
    area_adjustment = area_based_calculate_area_adjustment(tier_data, target_area)
    
    # B. Furnishing adjustment
    furnishing_adjustment = area_based_get_furnishing_adjustment(target_furnishing, furnishing_used, label_encoders)
    
    # C. Floor adjustment
    floor_adjustment = area_based_calculate_floor_adjustment(target_floor, target_total_floors)
    
    # Step 7: Calculate final estimate
    base_rent = baseline_rent_per_sqft * target_area
    adjusted_rent = base_rent * area_adjustment * furnishing_adjustment * floor_adjustment
    
    # Step 8: Calculate confidence and range
    confidence = selected_tier['confidence_base']
    if properties_count < 5:
        confidence -= 15
    
    range_percentage = area_based_get_range_percentage(selected_tier['name'])
    lower_bound = adjusted_rent * (1 - range_percentage/100)
    upper_bound = adjusted_rent * (1 + range_percentage/100)
    
    return {
        'estimated_rent': round(adjusted_rent, 0),
        'lower_bound': round(lower_bound, 0),
        'upper_bound': round(upper_bound, 0),
        'confidence': confidence,
        'tier_used': selected_tier['name'],
        'properties_count': properties_count,
        'baseline_rent_per_sqft': round(baseline_rent_per_sqft, 2),
        'furnishing_baseline': furnishing_used,
        'adjustments': {
            'area_adjustment': round((area_adjustment - 1) * 100, 1),
            'furnishing_adjustment': round((furnishing_adjustment - 1) * 100, 1),
            'floor_adjustment': round((floor_adjustment - 1) * 100, 1)
        },
        'explanation': f"Based on {properties_count} properties in '{selected_tier['name']}' category"
    }

def area_based_remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outlier properties based on rent per sqft"""
    if len(df) < 3 or 'total_rent' not in df.columns or 'builtup_area' not in df.columns:
        return df
    
    # Calculate rent per sqft
    df = df.copy()
    df['rent_per_sqft'] = df['total_rent'] / df['builtup_area']
    
    # Remove outliers (2 standard deviations from median)
    median_rent_psf = df['rent_per_sqft'].median()
    std_rent_psf = df['rent_per_sqft'].std()
    
    lower_bound = median_rent_psf - 2 * std_rent_psf
    upper_bound = median_rent_psf + 2 * std_rent_psf
    
    filtered_df = df[
        (df['rent_per_sqft'] >= lower_bound) & 
        (df['rent_per_sqft'] <= upper_bound)
    ]
    
    return filtered_df.drop('rent_per_sqft', axis=1)

def area_based_calculate_baseline_rent_per_sqft(df: pd.DataFrame, label_encoders: Optional[Dict] = None) -> Dict:
    """Calculate baseline rent per sqft prioritizing semi-furnished properties"""

    if 'furnishing' not in df.columns:
        # No furnishing data available

        total_rent = df['total_rent'].sum()
        total_area = df['builtup_area'].sum()
        baseline_rent_per_sqft = total_rent / total_area if total_area > 0 else 0
        
        return {
            'baseline_rent_per_sqft': baseline_rent_per_sqft,
            'furnishing_used': 'mixed',
            'properties_count': len(df)
        }
    
    # Get furnishing categories
    furnishing_priorities = ['Semi-Furnished', 'Unfurnished', 'Furnished']

    # Try to find the best furnishing category with >= 3 properties
    for furnishing_type in furnishing_priorities:
        # Handle both encoded and string values

        if label_encoders and 'furnishing' in label_encoders:

            try:
                # Try to find the encoded value
                furnishing_encoded = label_encoders['furnishing'].transform([furnishing_type])[0]
                furnishing_df = df[df['furnishing'] == furnishing_encoded]
            except:
                # Try string matching
                furnishing_df = df[df['furnishing'].astype(str).str.lower().str.contains(furnishing_type.lower(), na=False)]
        else:
            # Direct string matching

            furnishing_df = df[df['furnishing'].astype(str).str.lower().str.contains(furnishing_type.lower(), na=False)]

        if len(furnishing_df) >= 3:
            total_rent = furnishing_df['total_rent'].sum()
            total_area = furnishing_df['builtup_area'].sum()
            baseline_rent_per_sqft = total_rent / total_area if total_area > 0 else 0
            
            return {
                'baseline_rent_per_sqft': baseline_rent_per_sqft,
                'furnishing_used': furnishing_type,
                'properties_count': len(furnishing_df)
            }
    
    # If no single furnishing type has enough data, combine all
    total_rent = df['total_rent'].sum()
    total_area = df['builtup_area'].sum()
    baseline_rent_per_sqft = total_rent / total_area if total_area > 0 else 0

    return {
        'baseline_rent_per_sqft': baseline_rent_per_sqft,
        'furnishing_used': 'mixed',
        'properties_count': len(df)
    }

def area_based_calculate_area_adjustment(tier_data: pd.DataFrame, target_area: float) -> float:
    """Calculate area-based adjustment factor"""
    if 'builtup_area' not in tier_data.columns or len(tier_data) == 0:
        return 1.0
    
    avg_area = tier_data['builtup_area'].mean()
    
    if target_area > 1.3 * avg_area:
        # Larger units typically have lower per-sqft rates
        return 0.97  # 3% discount
    elif target_area < 0.7 * avg_area:
        # Smaller units typically have higher per-sqft rates
        return 1.03  # 3% premium
    else:
        return 1.0  # No adjustment

def area_based_get_furnishing_adjustment(target_furnishing, baseline_furnishing, label_encoders: Optional[Dict] = None) -> float:
    """Get furnishing adjustment factor using encoded values"""
    
    # If we have label encoders, work with encoded values
    if label_encoders and 'furnishing' in label_encoders:
        
        # Convert to encoded values if they're strings
        if isinstance(target_furnishing, str):
            try:
                target_encoded = label_encoders['furnishing'].transform([target_furnishing])[0]
            except:
                target_encoded = target_furnishing
        else:
            target_encoded = target_furnishing
            
        if isinstance(baseline_furnishing, str):
            try:
                baseline_encoded = label_encoders['furnishing'].transform([baseline_furnishing])[0]
            except:
                baseline_encoded = baseline_furnishing
        else:
            baseline_encoded = baseline_furnishing
        
        # Define adjustment factors by encoded value
        # You'll need to check what your encoded values are:
        # 0 = unfurnished, 1 = furnished, 2 = semi furnished (example)
        encoded_factors = {
            # Replace these with your actual encoded values:
            2: 0.9,  # unfurnished
            0: 1.1,  # furnished  
            1: 1.0,  # semi furnished
            # Add more if needed
        }
        
        target_factor = encoded_factors.get(target_encoded, 1.0)
        baseline_factor = encoded_factors.get(baseline_encoded, 1.0)
        
        return target_factor / baseline_factor
    
    # Fallback to string matching if no encoders
    else:
        # Your existing string logic here as backup
        pass

def area_based_calculate_floor_adjustment(floor: int, total_floors: int) -> float:
    """Calculate floor-based adjustment factor"""
    if total_floors <= 1:
        return 1.0
    
    floor_ratio = floor / total_floors
    
    if floor == 1:  # Ground floor
        return 0.98  # 2% discount
    elif floor == total_floors:  # Top floor
        return 1.03  # 3% premium
    elif floor_ratio > 0.8:  # High floors
        return 1.02  # 2% premium
    else:
        return 1.0  # No adjustment

def area_based_get_range_percentage(tier_name: str) -> float:
    """Get confidence range percentage based on tier"""
    range_map = {
        'Same Society, Same BHK': 5,
        'Same Society, Different BHK': 8,
        'Same Locality, Same BHK': 10,
        'Same Locality, Different BHK': 15,
        'City Average': 20
    }
    return range_map.get(tier_name, 15)

def area_based_calculate_city_average_estimate(df: pd.DataFrame, property_data: Dict) -> Dict:
    """Fallback to city-wide average when insufficient tier data"""
    
    target_area = property_data.get('builtup_area', 1000)
    
    if 'total_rent' not in df.columns or 'builtup_area' not in df.columns:
        return {
            'estimated_rent': 0,
            'lower_bound': 0,
            'upper_bound': 0,
            'confidence': 30,
            'tier_used': 'City Average',
            'properties_count': 0,
            'baseline_rent_per_sqft': 0,
            'furnishing_baseline': 'unknown',
            'adjustments': {'area_adjustment': 0, 'furnishing_adjustment': 0, 'floor_adjustment': 0},
            'explanation': 'Insufficient comparable properties, using city-wide average'
        }
    
    # Calculate city average rent per sqft
    total_rent = df['total_rent'].sum()
    total_area = df['builtup_area'].sum()
    city_avg_rent_per_sqft = total_rent / total_area if total_area > 0 else 0
    
    estimated_rent = city_avg_rent_per_sqft * target_area
    
    # Apply basic furnishing adjustment
    target_furnishing = str(property_data.get('furnishing', '')).lower()
    if 'furnished' in target_furnishing and 'semi' not in target_furnishing:
        estimated_rent *= 1.1
    elif 'unfurnished' in target_furnishing:
        estimated_rent *= 0.9
    
    range_pct = 20  # 20% range for city average
    lower_bound = estimated_rent * 0.8
    upper_bound = estimated_rent * 1.2
    
    return {
        'estimated_rent': round(estimated_rent, 0),
        'lower_bound': round(lower_bound, 0),
        'upper_bound': round(upper_bound, 0),
        'confidence': 40,
        'tier_used': 'City Average',
        'properties_count': len(df),
        'baseline_rent_per_sqft': round(city_avg_rent_per_sqft, 2),
        'furnishing_baseline': 'city_average',
        'adjustments': {
            'area_adjustment': 0,
            'furnishing_adjustment': 10 if 'furnished' in target_furnishing else (-10 if 'unfurnished' in target_furnishing else 0),
            'floor_adjustment': 0
        },
        'explanation': f'Based on city-wide average from {len(df)} properties (insufficient comparable data)'
    }
