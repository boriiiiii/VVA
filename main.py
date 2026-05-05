import numpy as np
import pandas as pd
import plotly.graph_objects as go
import folium

from data_loader import load_raw_data, scrape_circuit_info
from preprocessing import (
    preprocess_circuits, preprocess_drivers, preprocess_standings,
    preprocess_driver_standings, preprocess_lap_times, preprocess_pit_stops,
    preprocess_qualifying, preprocess_results, preprocess_sprint_results,
    preprocess_weather,
)
from features import (
    compute_constructor_experience, compute_driver_experience,
    compute_driver_age, compute_driver_wins, compute_constructor_wins,
    compute_driver_constructor_experience, compute_dnf_score, build_formula1,
)
from model import train_model, evaluate_model, save_model
from predict import prediction


def main():
    """Run the full F1 pipeline: load → preprocess → features → train → predict."""

    # --- Load raw data ---
    data = load_raw_data()
    df_circuits              = data['circuits']
    df_constructors          = data['constructors']
    df_constructor_results   = data['constructor_results']
    df_constructor_standings = data['constructor_standings']
    df_driver_standings      = data['driver_standings']
    df_drivers               = data['drivers']
    df_lap_times             = data['lap_times']
    df_pit_stops             = data['pit_stops']
    df_qualifying            = data['qualifying']
    df_results               = data['results']
    df_seasons               = data['seasons']
    df_sprint_results        = data['sprint_results']
    df_status                = data['status']
    df_races                 = data['races']
    df_weather               = data['weather']

    # --- Scrape Wikipedia for circuit Length and Turns ---
    df_circuits = scrape_circuit_info(df_circuits)

    # --- Preprocessing ---
    df_circuits              = preprocess_circuits(df_circuits)
    df_drivers               = preprocess_drivers(df_drivers)
    df_constructor_standings = preprocess_standings(df_constructor_standings, df_constructors, df_races)
    df_driver_standings      = preprocess_driver_standings(df_driver_standings, df_drivers)
    df_lap_times             = preprocess_lap_times(df_lap_times, df_races)
    df_pit_stops             = preprocess_pit_stops(df_pit_stops, df_races)
    df_qualifying            = preprocess_qualifying(df_qualifying)
    df_results               = preprocess_results(df_results, df_races)
    df_sprint_results        = preprocess_sprint_results(df_sprint_results, df_races)
    df_weather               = preprocess_weather(df_weather, df_races)

    # --- Exploratory analysis: constructor points vs wins ---
    df_constructor_points = (
        df_constructor_standings.groupby('name')['points'].sum()
        .sort_values(ascending=False).reset_index()
    )
    df_constructor_wins_agg = (
        df_constructor_standings[df_constructor_standings['position'] == 1]
        .groupby('name').size().reset_index().rename(columns={0: 'Wins'})
    )
    df_const = pd.merge(df_constructor_points, df_constructor_wins_agg, on='name')

    fig = go.Figure()
    for team in df_const['name'].unique():
        df_team = df_const[df_const['name'] == team]
        fig.add_trace(go.Scatter(x=df_team['Wins'], y=df_team['points'], mode='lines+markers', name=team))
    fig.update_layout(
        title_text='<b>Constructor Points vs Wins</b>',
        titlefont=dict(family='Arial, sans-serif', size=30),
        title_x=0.5, xaxis_title="Wins", yaxis_title="Points"
    )

    correlation = df_const['points'].corr(df_const['Wins'])
    print(f"constructor win correaltion: {correlation}")

    # --- Exploratory analysis: driver points vs wins ---
    df_driver_points = (
        df_driver_standings.groupby('Name')['points'].sum()
        .sort_values(ascending=False).reset_index()
    )
    df_driver_wins_agg = (
        df_driver_standings[df_driver_standings['position'] == 1]
        .groupby('Name').size().reset_index().rename(columns={0: 'Wins'})
    )
    df_drv = pd.merge(df_driver_points, df_driver_wins_agg, on='Name')

    fig = go.Figure()
    for team in df_drv['Name'].unique():
        df_team = df_drv[df_drv['Name'] == team]
        fig.add_trace(go.Scatter(x=df_team['Wins'], y=df_team['points'], mode='lines+markers', name=team))
    fig.update_layout(
        title_text='<b>Driver Points vs Wins</b>',
        titlefont=dict(family='Arial, sans-serif', size=30),
        title_x=0.5, xaxis_title="Wins", yaxis_title="Points"
    )

    correlation = df_drv['points'].corr(df_drv['Wins'])
    print(f"pilots win correaltion: {correlation}")

    # --- Grid position vs win rate ---
    df_grid_wins = df_results[['raceId', 'driverId', 'grid', 'position']].copy()
    df_grid_wins['Win'] = df_grid_wins['position'].apply(lambda x: 1 if x == 1 else 0)
    mean_wins_by_grid = (df_grid_wins.groupby('grid')['Win'].mean() * 100).round(2)
    print(mean_wins_by_grid)

    # --- Rain analysis ---
    threshold = 0.5
    rainy_races = df_weather[df_weather['Rainfall'] > threshold]
    percentage_rain = round((len(rainy_races) / len(df_weather)) * 100, 2)
    num_rainy_races = len(rainy_races)
    print(f'rainy races: {rainy_races}')
    print(f'number of races: {len(df_weather)}')
    print(f"Percent of races when it rained more than 50% of the race: {percentage_rain}%")
    print(f"Number of races where it rained more than 50% of the race: {num_rainy_races}")

    races_in_rain = df_weather[df_weather['Rainfall'] > 0]
    df_rain_grid_wins = df_grid_wins[df_grid_wins['raceId'].isin(races_in_rain['raceId'])]
    mean_wins_by_grid = (df_rain_grid_wins.groupby('grid')['Win'].mean() * 100).round(2)

    # --- Circuit map ---
    m = folium.Map(location=[20, 0], zoom_start=2)
    for idx, row in df_circuits.iterrows():
        folium.Marker([row['lat'], row['lng']], popup=f"{row['location']}, {row['country']}").add_to(m)
    # print(m)

    # --- Feature engineering ---
    df_constructor_experience  = compute_constructor_experience(df_results)
    df_driver_experience       = compute_driver_experience(df_results)
    df_driver_age              = compute_driver_age(df_results, df_drivers)
    df_driver_wins             = compute_driver_wins(df_results)
    df_constructor_wins        = compute_constructor_wins(df_constructor_standings)
    df_driver_constructor_exp  = compute_driver_constructor_experience(df_results)
    df_finish                  = compute_dnf_score(df_results, df_status)

    formula_1 = build_formula1(
        df_results, df_circuits,
        df_constructor_experience, df_driver_experience,
        df_driver_age, df_driver_wins, df_constructor_wins,
        df_driver_constructor_exp, df_finish
    )

    # --- Model training and evaluation ---
    formula1_predict, X, y, X_test, y_test = train_model(formula_1)
    evaluate_model(formula1_predict, X, y, X_test, y_test)
    save_model(formula1_predict)

    # --- Sample predictions for Marina Bay ---
    drivers = [
        "Lando Norris",
        "Max Verstappen",
        "Lewis Hamilton",
        "George Russell",
        "Oscar Piastri",
        "Nico Hülkenberg",
        "Fernando Alonso",
        "Yuki Tsunoda",
        "Charles Leclerc",
        "Carlos Sainz",
        "Alexander Albon",
        "Sergio Pérez",
        "Kevin Magnussen",
        "Esteban Ocon",
        "Daniel Ricciardo",
        "Lance Stroll",
        "Pierre Gasly",
        "Valtteri Bottas",
        "Guanyu Zhou"
    ]
    grids = list(range(1, 20))
    circuit_loc = 'Marina Bay'

    predictions = []
    for driver_name, grid in zip(drivers, grids):
        pred, prob = prediction(driver_name, grid, circuit_loc, formula1_predict, formula_1, df_drivers, df_circuits)
        if pred in [1, 2, 3]:
            predictions.append({
                'Driver Name ': driver_name,
                'Grid': grid,
                'Prediction': pred,
                'Probability': np.max(prob)
            })

    print(predictions)


if __name__ == '__main__':
    main()
