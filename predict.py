import numpy as np
import pandas as pd


def prediction(driver_name, grid, circuit_loc, formula1_predict, formula_1, df_drivers, df_circuits):
    """Predict podium outcome for a driver given their grid position and circuit."""
    driver = df_drivers.loc[df_drivers['Name'] == driver_name, 'driverId'].iloc[0]

    # Use the driver's most recent race as baseline for their current stats
    input_data = formula_1[formula_1['driverId'] == driver].sort_values(by='date', ascending=False).iloc[0]
    circuit_data = df_circuits[df_circuits['location'] == circuit_loc].iloc[0]

    features = {
        'driverId': input_data['driverId'],
        'constructorId': input_data['constructorId'],
        'grid': grid,
        'laps': circuit_data['laps'],
        'circuitId': circuit_data['circuitId'],
        'Length': circuit_data['Length'],
        'Turns': circuit_data['Turns'],
        'Constructor Experience': input_data['Constructor Experience'],
        'Driver Experience': input_data['Driver Experience'],
        'Driver Age': input_data['Driver Age'],
        'Driver Wins': input_data['Driver Wins'],
        'Constructor Wins': input_data['Constructor Wins'],
        'Driver Constructor Experience': input_data['Driver Constructor Experience'],
        'DNF Score': input_data['DNF Score'],
        'prev_position': input_data['prev_position']
    }
    features = pd.DataFrame([features])

    return formula1_predict.predict(features), formula1_predict.predict_proba(features)
