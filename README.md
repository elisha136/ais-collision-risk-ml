# AIS Collision Risk Prediction using Machine Learning

A Random Forest-based collision risk prediction system for maritime vessels using AIS (Automatic Identification System) data. This project analyzes ship trajectories and predicts collision risk levels.
##  Project Overview

This project implements a machine learning model that:
- Analyzes AIS trajectory data from vessels in the Bergen, Norway region
- Extracts 25 features related to collision avoidance (DCPA, TCPA, relative bearing, etc.)
- Predicts collision risk levels (0-3) using a Random Forest classifier
- 
## Dataset

- **Source:** PostgreSQL database with AIS position data
- **Location:** Bergen, Norway maritime area
- **Time Period:** November 23 - December 7, 2024 (14 days)
- **Vessels:** 30 tracked vessels
- **Records:** 1.3M AIS position records
- **Encounters:** 317,367 vessel pair encounters analyzed
##  Model Details

### Architecture
- **Algorithm:** Random Forest Classifier
- **Trees:** 300 decision trees
- **Max Depth:** 15 levels
- **Features:** 25 engineered features
- **Output:** 4 risk levels (0=No Risk, 1=Low, 2=Medium, 3=High)


### Top 5 Most Important Features
1. **TCPA** (30.7%) - Time to Closest Point of Approach
2. **DCPA** (23.5%) - Distance at Closest Point of Approach
3. **Relative Speed** (11.3%) - Speed difference between vessels
4. **Distance** (11.3%) - Current distance between vessels
5. **Target Speed** (4.9%) - Speed of the other vessel

## Risk Level Classification 

| Risk Level | DCPA Threshold | TCPA Threshold | Action Required |
|------------|----------------|----------------|-----------------|
| **0 - No Risk** | > 2.0 nm | > 30 minutes | Normal navigation |
| **1 - Low Risk** | 1.0 - 2.0 nm | 20 - 30 minutes | Monitor situation |
| **2 - Medium Risk** | 0.5 - 1.0 nm | 10 - 20 minutes | Prepare to maneuver |
| **3 - High Risk** | < 0.5 nm | < 10 minutes | Immediate action required |


## Features


The model uses **25 engineered features** across 6 categories:

### 1. Core Collision Avoidance (4 features)
- `dcpa`, `tcpa`, `distance`, `relative_speed`

### 2. Geometric Features (3 features)
- `relative_bearing`, `aspect_angle`, `cpa_bearing`

### 3. Vessel Motion (6 features)
- `sog_1`, `sog_2`, `cog_1`, `cog_2`, `speed_ratio`, `course_difference`

### 4. COLREGS Encounter Types (4 features)
- `is_head_on`, `is_crossing`, `is_overtaking`, `give_way_status`

### 5. Dynamic Features (4 features)
- `rate_of_bearing_change`, `distance_rate_of_change`, `avg_speed_last_5min`, `course_stability`

### 6. Temporal Features (4 features)
- `hour_of_day`, `day_of_week`, `time_since_last_update`, `prediction_horizon`






## Data Extraction
   - Query AIS position data from PostgreSQL database
   - Filter by geographic area (Bergen, Norway)
   - Select 30 vessels over 14 day period

## Coordinate Conversion
   - Convert lat/lon to NED (North-East-Down) coordinate system
   - Use MSS (Marine Systems Simulator) flat-earth approximation
   - Reference: Fossen (2011) - Handbook of Marine Craft Hydrodynamics

## Encounter Detection
   - Sliding time window approach (5-minute windows, 1-minute slide)
   - Detect all vessel pairs within detection range
   - Calculate encounter geometry and dynamics

## Feature Engineering
   - Extract 25 features per encounter
   - Calculate DCPA/TCPA using relative motion equations
   - Determine COLREGS encounter types (head-on, crossing, overtaking)
   - Compute dynamic features (rate of change, stability metrics)

## Risk Labeling
   - Assign risk levels (0-3) based on DCPA/TCPA thresholds
   - Based on maritime collision avoidance research (Hwang 2002, MDPI 2024)

## Model Training
   - Train/validation/test split: 70/15/15
   - Random Forest with 300 trees, max depth 15
   - Feature scaling using StandardScaler
  


