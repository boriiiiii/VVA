import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

# Load your model
model = load("model/formula1_model.joblib")

# Load your data
df_formula1 = pd.read_csv("data/formula1.csv")
df_circuits = pd.read_csv("data/circuits.csv")
df_drivers = pd.read_csv("data/drivers.csv")


def prediction(driver_name, grid, circuit_loc):
    # Your prediction logic here
    # Use the loaded model for prediction
    driver = df_drivers.loc[df_drivers['Name'] == driver_name, 'driverId'].iloc[0]
    circuit = df_circuits.loc[df_circuits['location'] == circuit_loc, ['circuitId', 'laps']].iloc[0]

    input_data = df_formula1[df_formula1['driverId'] == driver].sort_values(by='date', ascending=False).iloc[0]
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

    return model.predict(features), model.predict_proba(features)


def main():
    st.title('F1 Race Prediction')

    # front end elements of the web pagea
    html_temp = """ 
        <div style ="background-color:yellow;padding:13px"> 
        <h1 style ="color:black;text-align:center;">F1 Race Prediction</h1> 
        </div> 
        """
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)
    default_drivers = [
        "Charles Leclerc",
        "Oscar Piastri",
        "Carlos Sainz",
        "Sergio Pérez",
        "George Russell",
        "Max Verstappen",
        "Fernando Alonso",
        "Alexander Albon",
        "Oliver Bearman",
        "Yuki Tsunoda",
        "Nico Hülkenberg",
        "Lance Stroll",
        "Daniel Ricciardo",
        "Lando Norris",
        "Valtteri Bottas",
        "Guanyu Zhou",
        "Pierre Gasly",
        "Lewis Hamilton",
        "Esteban Ocon"
    ]
    default_grid = '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19'
    default_circuit = 'Baku'

    # Input for driver names
    all_drivers = ['Max Verstappen', 'Charles Leclerc', 'George Russell', 'Carlos Sainz', 'Sergio Pérez',
                   'Fernando Alonso', 'Lando Norris', 'Oscar Piastri', 'Lewis Hamilton', 'Nico Hülkenberg',
                   'Yuki Tsunoda', 'Lance Stroll', 'Alexander Albon', 'Daniel Ricciardo', 'Kevin Magnussen',
                   'Valtteri Bottas', 'Logan Sargeant', 'Esteban Ocon', 'Pierre Gasly', 'Oliver Bearman', 'Guanyu Zhou']
    selected_drivers = st.multiselect('Select drivers:', all_drivers, default=default_drivers)
    # Output selected drivers
    st.write('Selected drivers:', selected_drivers)

    # Input for grid positions
    grid_positions = st.text_input('Enter grid positions for selected drivers (comma separated):', value=default_grid)
    grid_positions = list(map(int, (gp for gp in grid_positions.split(',') if gp)))

    # Input for circuit location
    circuit_loc = st.text_input('Enter circuit location:', value=default_circuit)

    predictions = []

    # Function Call
    if st.button('Predict'):
        for driver_name, grid in zip(selected_drivers, grid_positions):
            pred, prob = prediction(driver_name, grid, circuit_loc)
            if pred in [1, 2, 3]:
                predictions.append({
                    'Driver Name': driver_name,
                    'Grid': grid,
                    'Prediction': pred,
                    'Probability': np.max(prob)
                })

        st.success("Drivers on Podium: \n {}".format(predictions))


if __name__ == '__main__':
    main()
