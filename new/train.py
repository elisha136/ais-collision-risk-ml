"""
AIS Collision Risk Prediction - CLEAN MODEL ONLY
=================================================
DCPA and TCPA removed from input features to fix data leakage.

"""

# =============================================================================
# DEPENDENCIES
# =============================================================================
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 70)
print("AIS COLLISION RISK - CLEAN MODEL (NO DATA LEAKAGE)")
print("=" * 70)
print()

# =============================================================================
# CONFIGURATION
# =============================================================================
DB_URL = "postgresql://postgres:changeme-strong-pass@localhost:5432/adimalara"
NUM_VESSELS = 10
TIME_WINDOW_DAYS = 14

print(f"Configuration: {NUM_VESSELS} vessels, {TIME_WINDOW_DAYS} days")
print()

# =============================================================================
# DATABASE CONNECTION
# =============================================================================
print(" Connecting to database...")
engine = create_engine(DB_URL)

try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version();"))
        version = result.fetchone()[0]
        print(f"✓ Connected: {version.split(',')[0]}")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    exit(1)
print()

# =============================================================================
# DISCOVER DATA RANGE
# =============================================================================
print(" Discovering data range...")

discovery_query = text("""
    SELECT 
        MIN(timestamp) as earliest, MAX(timestamp) as latest,
        MIN(latitude) as min_lat, MAX(latitude) as max_lat,
        MIN(longitude) as min_lon, MAX(longitude) as max_lon,
        COUNT(*) as total_records, COUNT(DISTINCT mmsi) as unique_vessels
    FROM ais1_position
""")

data_range = pd.read_sql_query(discovery_query, engine)
print(f"  Total records: {data_range['total_records'][0]:,}")

earliest = data_range['earliest'][0]
latest = data_range['latest'][0]
total_days = (latest - earliest).days
middle_date = earliest + timedelta(days=total_days // 2)

START_TIME = middle_date - timedelta(days=TIME_WINDOW_DAYS // 2)
END_TIME = middle_date + timedelta(days=TIME_WINDOW_DAYS // 2)

LAT_MIN, LAT_MAX = float(data_range['min_lat'][0]), float(data_range['max_lat'][0])
LON_MIN, LON_MAX = float(data_range['min_lon'][0]), float(data_range['max_lon'][0])

print(f"  Query window: {START_TIME.date()} to {END_TIME.date()}")
print()

# =============================================================================
# SELECT VESSELS
# =============================================================================
print(f"Selecting {NUM_VESSELS} most active vessels...")

vessel_query = text("""
    SELECT mmsi, COUNT(*) as record_count
    FROM ais1_position
    WHERE timestamp BETWEEN :start_time AND :end_time
      AND latitude BETWEEN :lat_min AND :lat_max
      AND longitude BETWEEN :lon_min AND :lon_max
    GROUP BY mmsi HAVING COUNT(*) > 50
    ORDER BY record_count DESC LIMIT :num_vessels
""")

vessels_df = pd.read_sql_query(vessel_query, engine, params={
    'start_time': START_TIME, 'end_time': END_TIME,
    'lat_min': LAT_MIN, 'lat_max': LAT_MAX,
    'lon_min': LON_MIN, 'lon_max': LON_MAX,
    'num_vessels': NUM_VESSELS
})
MMSI_LIST = vessels_df['mmsi'].tolist()
print(f"Found {len(vessels_df)} vessels, {vessels_df['record_count'].sum():,} records")
print()

# =============================================================================
# LOAD DATA
# =============================================================================
print(" Loading trajectory data...")

main_query = text("""
    SELECT mmsi, timestamp, latitude, longitude, sog, cog, true_heading, navigational_status
    FROM ais1_position
    WHERE mmsi = ANY(:mmsi_list) AND timestamp BETWEEN :start_time AND :end_time
    ORDER BY mmsi, timestamp
""")

df = pd.read_sql_query(main_query, engine, params={
    'mmsi_list': MMSI_LIST, 'start_time': START_TIME, 'end_time': END_TIME
})
print(f" Loaded {len(df):,} records")
print()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def latlon_to_ned(lat, lon, lat_ref, lon_ref):
    R_EARTH = 6371000.0
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    lat_ref_rad, lon_ref_rad = np.radians(lat_ref), np.radians(lon_ref)
    north = R_EARTH * (lat_rad - lat_ref_rad)
    east = R_EARTH * np.cos(lat_ref_rad) * (lon_rad - lon_ref_rad)
    return north, east

def calculate_dcpa_tcpa(pos_a, vel_a, pos_b, vel_b):
    r_x, r_y = pos_b[0] - pos_a[0], pos_b[1] - pos_a[1]
    v_rel_x, v_rel_y = vel_b[0] - vel_a[0], vel_b[1] - vel_a[1]
    v_rel_mag = np.sqrt(v_rel_x**2 + v_rel_y**2)
    
    if v_rel_mag < 0.01:
        return np.sqrt(r_x**2 + r_y**2), 86400.0
    
    tcpa = -(r_x * v_rel_x + r_y * v_rel_y) / (v_rel_mag**2)
    tcpa = max(0, min(tcpa, 86400.0))
    
    pos_a_cpa = (pos_a[0] + vel_a[0] * tcpa, pos_a[1] + vel_a[1] * tcpa)
    pos_b_cpa = (pos_b[0] + vel_b[0] * tcpa, pos_b[1] + vel_b[1] * tcpa)
    dcpa = np.sqrt((pos_b_cpa[0] - pos_a_cpa[0])**2 + (pos_b_cpa[1] - pos_a_cpa[1])**2)
    return dcpa, tcpa

def calculate_bearing(from_pos, to_pos):
    delta_north, delta_east = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
    bearing_deg = np.degrees(np.arctan2(delta_east, delta_north))
    return bearing_deg + 360 if bearing_deg < 0 else bearing_deg

def normalize_angle_diff(angle_diff):
    angle_diff = abs(angle_diff)
    return 360 - angle_diff if angle_diff > 180 else angle_diff

def classify_encounter_type(cog_a, cog_b, bearing_a_to_b, bearing_b_to_a):
    rel_bearing_a = normalize_angle_diff(bearing_a_to_b - cog_a)
    rel_bearing_b = normalize_angle_diff(bearing_b_to_a - cog_b)
    course_diff = normalize_angle_diff(cog_a - cog_b)
    
    is_head_on = course_diff < 10 and rel_bearing_a < 10 and rel_bearing_b < 10
    is_overtaking = rel_bearing_b > 157.5 or rel_bearing_a > 157.5
    is_crossing = not is_head_on and not is_overtaking
    
    give_way_status = 0
    if is_crossing:
        give_way_status = 1 if 0 < rel_bearing_a < 180 else 2
    elif is_overtaking:
        give_way_status = 1 if rel_bearing_b > 157.5 else 2
    
    return {
        'is_head_on': int(is_head_on),
        'is_crossing': int(is_crossing),
        'is_overtaking': int(is_overtaking),
        'give_way_status': give_way_status
    }

def detect_encounters(df, window_size_minutes=5, slide_minutes=1, distance_threshold_nm=5):
    encounters = []
    distance_threshold_m = distance_threshold_nm * 1852
    
    start_time, end_time = df['timestamp'].min(), df['timestamp'].max()
    current_time = start_time
    window_delta = timedelta(minutes=window_size_minutes)
    slide_delta = timedelta(minutes=slide_minutes)
    
    while current_time < end_time:
        window_end = current_time + window_delta
        window_data = df[(df['timestamp'] >= current_time) & (df['timestamp'] < window_end)]
        vessels_in_window = window_data['mmsi'].unique()
        
        for i, mmsi_a in enumerate(vessels_in_window):
            for mmsi_b in vessels_in_window[i+1:]:
                va = window_data[window_data['mmsi'] == mmsi_a]
                vb = window_data[window_data['mmsi'] == mmsi_b]
                if len(va) == 0 or len(vb) == 0:
                    continue
                va, vb = va.iloc[-1], vb.iloc[-1]
                distance = np.sqrt((va['north'] - vb['north'])**2 + (va['east'] - vb['east'])**2)
                if distance <= distance_threshold_m:
                    encounters.append({'timestamp': current_time, 'mmsi_1': mmsi_a, 'mmsi_2': mmsi_b, 'distance': distance})
        
        current_time += slide_delta
        if len(encounters) % 500 == 0 and len(encounters) > 0:
            print(f"  Found {len(encounters)} encounters...", end='\r')
    
    print(f"✓ Detected {len(encounters)} encounters" + " " * 20)
    return pd.DataFrame(encounters)

def extract_features(encounter_row, df):
    """Extract 23 features (NO DCPA/TCPA to avoid data leakage)."""
    timestamp = encounter_row['timestamp']
    mmsi_1, mmsi_2 = encounter_row['mmsi_1'], encounter_row['mmsi_2']
    
    v1_data = df[(df['mmsi'] == mmsi_1) & (df['timestamp'] <= timestamp)].tail(10)
    v2_data = df[(df['mmsi'] == mmsi_2) & (df['timestamp'] <= timestamp)].tail(10)
    
    if len(v1_data) == 0 or len(v2_data) == 0:
        return None
    
    v1, v2 = v1_data.iloc[-1], v2_data.iloc[-1]
    pos_1, pos_2 = (v1['north'], v1['east']), (v2['north'], v2['east'])
    vel_1, vel_2 = (v1['v_north'], v1['v_east']), (v2['v_north'], v2['v_east'])
    
    # Calculate DCPA/TCPA for labeling ONLY (not as features)
    dcpa, tcpa = calculate_dcpa_tcpa(pos_1, vel_1, pos_2, vel_2)
    
    # Geometric features
    distance = encounter_row['distance']
    bearing_1_to_2 = calculate_bearing(pos_1, pos_2)
    bearing_2_to_1 = calculate_bearing(pos_2, pos_1)
    relative_bearing = normalize_angle_diff(bearing_1_to_2 - v1['cog'])
    aspect_angle = normalize_angle_diff(bearing_2_to_1 - v2['cog'])
    cpa_pos_1 = (pos_1[0] + vel_1[0] * tcpa, pos_1[1] + vel_1[1] * tcpa)
    cpa_pos_2 = (pos_2[0] + vel_2[0] * tcpa, pos_2[1] + vel_2[1] * tcpa)
    cpa_bearing = calculate_bearing(cpa_pos_1, cpa_pos_2)
    
    # Kinematic features
    relative_speed = np.sqrt((vel_2[0] - vel_1[0])**2 + (vel_2[1] - vel_1[1])**2)
    speed_ratio = v1['sog'] / (v2['sog'] + 0.01)
    course_difference = normalize_angle_diff(v1['cog'] - v2['cog'])
    
    rate_of_bearing_change = 0
    if len(v1_data) >= 2:
        prev_bearing = calculate_bearing((v1_data.iloc[-2]['north'], v1_data.iloc[-2]['east']), pos_2)
        time_diff = (v1['timestamp'] - v1_data.iloc[-2]['timestamp']).total_seconds() / 60
        rate_of_bearing_change = (bearing_1_to_2 - prev_bearing) / (time_diff + 0.01)
    
    # COLREGS encounter type
    encounter_type = classify_encounter_type(v1['cog'], v2['cog'], bearing_1_to_2, bearing_2_to_1)
    
    # Historical features
    distance_rate_of_change = 0
    if len(v1_data) >= 2:
        prev_dist = np.sqrt((v1_data.iloc[-2]['north'] - v2['north'])**2 + (v1_data.iloc[-2]['east'] - v2['east'])**2)
        time_diff = (v1['timestamp'] - v1_data.iloc[-2]['timestamp']).total_seconds()
        distance_rate_of_change = (distance - prev_dist) / (time_diff + 0.01)
    
    return {
        # Geometric (4) - NO DCPA/TCPA
        'distance': distance,
        'relative_bearing': relative_bearing,
        'aspect_angle': aspect_angle,
        'cpa_bearing': cpa_bearing,
        # Kinematic (8)
        'sog_1': v1['sog'], 'sog_2': v2['sog'],
        'cog_1': v1['cog'], 'cog_2': v2['cog'],
        'relative_speed': relative_speed,
        'speed_ratio': speed_ratio,
        'course_difference': course_difference,
        'rate_of_bearing_change': rate_of_bearing_change,
        # Encounter Type (4)
        'is_head_on': encounter_type['is_head_on'],
        'is_crossing': encounter_type['is_crossing'],
        'is_overtaking': encounter_type['is_overtaking'],
        'give_way_status': encounter_type['give_way_status'],
        # Temporal (4)
        'hour_of_day': timestamp.hour,
        'day_of_week': timestamp.weekday(),
        'time_since_last_update': (timestamp - v1['timestamp']).total_seconds(),
        'prediction_horizon': 300.0,
        # Historical (3)
        'avg_speed_last_5min': v1_data['sog'].mean(),
        'course_stability': v1_data['cog'].std(),
        'distance_rate_of_change': distance_rate_of_change,
        # For labeling only (removed before training)
        '_dcpa': dcpa, '_tcpa': tcpa
    }

def assign_risk_level(dcpa, tcpa):
    dcpa_nm, tcpa_min = dcpa / 1852.0, tcpa / 60.0
    if dcpa_nm < 0.5 or tcpa_min < 10: return 3
    elif dcpa_nm < 1.0 or tcpa_min < 20: return 2
    elif dcpa_nm < 2.0 or tcpa_min < 30: return 1
    else: return 0

# =============================================================================
# CONVERT COORDINATES
# =============================================================================
print(" Converting coordinates to NED...")

LAT_REF = (df['latitude'].min() + df['latitude'].max()) / 2
LON_REF = (df['longitude'].min() + df['longitude'].max()) / 2

df['north'], df['east'] = latlon_to_ned(df['latitude'].values, df['longitude'].values, LAT_REF, LON_REF)
df['v_north'] = df['sog'] * 0.514444 * np.cos(np.radians(df['cog']))
df['v_east'] = df['sog'] * 0.514444 * np.sin(np.radians(df['cog']))

print(f" Converted {len(df):,} positions")
print()

# =============================================================================
# DETECT ENCOUNTERS
# =============================================================================
print(" Detecting encounters...")
encounters_df = detect_encounters(df)
print()

# =============================================================================
# EXTRACT FEATURES
# =============================================================================
print("Extracting features (23 features, NO DCPA/TCPA)...")

features_list = []
for idx, encounter in encounters_df.iterrows():
    feat = extract_features(encounter, df)
    if feat is not None:
        features_list.append(feat)
    if (idx + 1) % 1000 == 0:
        print(f"  Processed {idx + 1}/{len(encounters_df)}...", end='\r')

features_df = pd.DataFrame(features_list)
print(f"Extracted {len(features_df):,} samples" + " " * 20)
print()

# =============================================================================
# ASSIGN LABELS
# =============================================================================
print(" Assigning risk labels...")

features_df['risk_level'] = features_df.apply(lambda r: assign_risk_level(r['_dcpa'], r['_tcpa']), axis=1)

print("Risk distribution:")
print(features_df['risk_level'].value_counts().sort_index())
print()

# =============================================================================
# PREPARE TRAINING DATA
# =============================================================================
print(" Preparing training data...")

# Remove labeling columns
X = features_df.drop(['risk_level', '_dcpa', '_tcpa'], axis=1)
y = features_df['risk_level']

print(f"Features: {X.shape[1]} (DCPA/TCPA excluded)")
print(f"Feature list: {list(X.columns)}")
print()

# Clean data
X = X.replace([np.inf, -np.inf], [1e10, -1e10]).fillna(X.median())

# Split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=RANDOM_SEED, stratify=y_temp)

print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print()

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# TRAIN MODEL
# =============================================================================
print(" Training Random Forest (300 trees)...")

model = RandomForestClassifier(
    n_estimators=300, max_depth=15,
    min_samples_split=10, min_samples_leaf=4,
    random_state=RANDOM_SEED, n_jobs=-1, oob_score=True, verbose=1
)
model.fit(X_train_scaled, y_train)
print(" Training complete")
print()

# =============================================================================
# EVALUATE
# =============================================================================
print(" Evaluating model...")

y_train_pred = model.predict(X_train_scaled)
y_val_pred = model.predict(X_val_scaled)
y_test_pred = model.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')

print()

print(f"  Training Accuracy:   {train_acc*100:.2f}%")
print(f"  Validation Accuracy: {val_acc*100:.2f}%")
print(f"  Test Accuracy:       {test_acc*100:.2f}%")
print(f"  Precision:           {precision*100:.2f}%")
print(f"  Recall:              {recall*100:.2f}%")
print(f"  F1-Score:            {f1*100:.2f}%")
print(f"  OOB Score:           {model.oob_score_*100:.2f}%")

print()

# Confusion Matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)
print()

# Feature Importance
print("Top 10 Features:")
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance_df.head(10).to_string(index=False))
print()

# =============================================================================
# SAVE OUTPUTS
# =============================================================================
print("Saving outputs...")

os.makedirs('outputs_clean', exist_ok=True)

joblib.dump(model, 'outputs_clean/collision_risk_model_CLEAN.pkl')
joblib.dump(scaler, 'outputs_clean/feature_scaler_CLEAN.pkl')

with open('outputs_clean/feature_names.txt', 'w') as f:
    for feat in X.columns:
        f.write(f"{feat}\n")

metadata = {
    'model_type': 'RandomForestClassifier',
    'n_features': len(X.columns),
    'features': list(X.columns),
    'n_estimators': 300,
    'max_depth': 15,
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'num_vessels': NUM_VESSELS,
    'test_accuracy': float(test_acc),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'oob_score': float(model.oob_score_),
    'note': 'DCPA and TCPA excluded from features to prevent data leakage'
}

with open('outputs_clean/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("Saved model and artifacts to outputs_clean/")
print()

# =============================================================================
# VISUALIZATIONS
# =============================================================================
print("Creating visualizations...")

os.makedirs('outputs_clean/plots', exist_ok=True)

# Feature Importance
plt.figure(figsize=(10, 6))
top_feat = importance_df.head(15)
plt.barh(range(len(top_feat)), top_feat['importance'], color='steelblue')
plt.yticks(range(len(top_feat)), top_feat['feature'])
plt.xlabel('Importance')
plt.title('Feature Importance (NO DCPA/TCPA)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('outputs_clean/plots/feature_importance.png', dpi=300)
plt.close()

# Confusion Matrix
plt.figure(figsize=(8, 6))
risk_classes = ['No Risk', 'Low', 'Medium', 'High']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=risk_classes, yticklabels=risk_classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix (Accuracy: {test_acc*100:.1f}%)')
plt.tight_layout()
plt.savefig('outputs_clean/plots/confusion_matrix.png', dpi=300)
plt.close()

print("Saved plots")
print()
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Model saved: outputs_clean/collision_risk_model_CLEAN.pkl")