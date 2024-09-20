import requests
from bs4 import BeautifulSoup
from io import StringIO

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

df_circuits = pd.read_csv("F1 Championship Archive/circuits.csv")
df_constructors = pd.read_csv("F1 Championship Archive/constructors.csv")
df_constructor_results = pd.read_csv("F1 Championship Archive/constructor_results.csv")
df_constructor_standings = pd.read_csv("F1 Championship Archive/constructor_standings.csv")
df_driver_standings = pd.read_csv("F1 Championship Archive/driver_standings.csv")
df_drivers = pd.read_csv("F1 Championship Archive/drivers.csv")
df_lap_times = pd.read_csv("F1 Championship Archive/lap_times.csv")
df_pit_stops = pd.read_csv("F1 Championship Archive/pit_stops.csv")
df_qualifying = pd.read_csv("F1 Championship Archive/qualifying.csv")
df_results = pd.read_csv("F1 Championship Archive/results.csv")
df_seasons = pd.read_csv("F1 Championship Archive/seasons.csv")
df_sprint_results = pd.read_csv("F1 Championship Archive/sprint_results.csv")
df_status = pd.read_csv("F1 Championship Archive/status.csv")
df_races = pd.read_csv("F1 Championship Archive/races.csv")
df_weather = pd.read_csv("F1 Championship Archive/F1_Meteo_2022_2024.csv")

len_turn_data = []


def length_turn(url, cId):
    response = requests.get(url)

    # Parse the HTML content
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the table element
    table = soup.find("table", class_="infobox")
    df = pd.read_html(StringIO(str(table)))[0]

    # If Length/Turns is not available set them to 0.
    try:
        length = df.loc[np.where(df == 'Length')[0][0]].iloc[1]
        turns = df.loc[np.where(df == 'Turns')[0][0]].iloc[1]
    except IndexError:
        length = '0.0000'
        turns = '0'

    # Append the List: [circuitId, Length, Turns]
    len_turn_data.append([cId, length, turns])


# Function to extract Length and Turns for each circuit according to circuitId
for cId, url in zip(df_circuits.circuitId, df_circuits.url):
    length_turn(url, cId)

# Convert the Length and Turns column in float and int types
df_len_turn = pd.DataFrame(data=len_turn_data, columns=["circuitId", "Length", "Turns"])

df_len_turn['Length'] = df_len_turn['Length'].str[:5].astype(float)
df_len_turn['Turns'] = df_len_turn['Turns'].str[:2].astype(int)

# Merge the Length & Turns dataframe to df_circuits according to circuitId
df_circuits = df_circuits.merge(df_len_turn, on='circuitId', how='left')

# Replaces '\N' values with NaN
df_circuits['alt'] = df_circuits['alt'].replace('\\N', np.nan)
df_circuits['alt'] = df_circuits['alt'].astype(float)


# Replacing missing values with mean of data
df_circuits['Length'] = df_circuits['Length'].replace(0,
                                                      df_circuits[df_circuits['Length'] != 0]['Length'].mean()).astype(
    float)
df_circuits['Turns'] = df_circuits['Turns'].replace(0, df_circuits[df_circuits['Turns'] != 0]['Turns'].mean()).astype(
    int)
df_circuits.loc[:, 'alt'] = df_circuits['alt'].fillna(df_circuits['alt'].mean()).astype(int)

df_circuits['laps'] = (305 / df_circuits['Length']).astype(int)

df_circuits.to_csv('circuits.csv', index=False)


df_constructor_standings = df_constructor_standings.merge(df_constructors[['constructorId', 'name']],
                                                          on='constructorId', how='left')

df_constructor_standings = df_constructor_standings.merge(df_races[['raceId', 'date']], on='raceId')

# No need for driver number
df_drivers.drop('number', axis=1, inplace=True)

# Adding driver 'Name' column instead of forename and surname
df_drivers['Name'] = df_drivers['forename'] + ' ' + df_drivers['surname']

# Adding code for each driver i.e, the first 3 letter of the surname
df_drivers['surname'] = df_drivers['surname'].str.replace(' ', '')


def replace_code(row):
    if row['code'] == '\\N':
        return row['surname'][:3].upper()
    else:
        return row['code']


df_drivers['code'] = df_drivers.apply(replace_code, axis=1)

# Dropping columns
df_drivers.drop(columns=['driverRef', 'forename', 'surname', 'url'], axis=1, inplace=True)

df_drivers.to_csv('drivers.csv', index=False)

df_driver_standings = df_driver_standings.merge(df_drivers[['driverId', 'Name']], on='driverId', how='left')

# Lap Times

df_lap_times = df_lap_times[df_lap_times['raceId'].isin(df_races['raceId'])]

# Assuming df_lap_times is your DataFrame and 'milliseconds' is the column with lap times
Q1 = df_lap_times['milliseconds'].quantile(0.25)
Q3 = df_lap_times['milliseconds'].quantile(0.75)
IQR = Q3 - Q1

# Define a filter for outliers
filter = (df_lap_times['milliseconds'] >= Q1 - 1.5 * IQR) & (df_lap_times['milliseconds'] <= Q3 + 1.5 * IQR)

# Create a new column 'outlier' that is True where the row is an outlier and False otherwise
df_lap_times['outlier'] = ~filter

# Plot the 'milliseconds' column, coloring by the 'outlier' column
# plt.figure(figsize=(10, 6))
# plt.scatter(df_lap_times.index, df_lap_times['milliseconds'], c=df_lap_times['outlier'])
# plt.title('Outliers in Lap Times')
# plt.xlabel('Index')
# plt.ylabel('Milliseconds')
# plt.show()

df_lap_times = df_lap_times[df_lap_times['milliseconds'] < 600000]

df_pit_stops

df_pit_stops = df_pit_stops[df_pit_stops['raceId'].isin(df_races['raceId'])]

import matplotlib.pyplot as plt

# Assuming df_pit_stops is your DataFrame and 'milliseconds' is the column with lap times
Q1 = df_pit_stops['milliseconds'].quantile(0.25)
Q3 = df_pit_stops['milliseconds'].quantile(0.75)
IQR = Q3 - Q1

# Define a filter for outliers
filter = (df_pit_stops['milliseconds'] >= Q1 - 1.5 * IQR) & (df_pit_stops['milliseconds'] <= Q3 + 1.5 * IQR)

# Create a new column 'outlier' that is True where the row is an outlier and False otherwise
df_pit_stops['outlier'] = ~filter

# Plot the 'milliseconds' column, coloring by the 'outlier' column
# plt.figure(figsize=(10, 6))
# plt.scatter(df_pit_stops.index, df_pit_stops['milliseconds'], c=df_pit_stops['outlier'])
# plt.title('Outliers in Pit Stop Times')
# plt.xlabel('Index')
# plt.ylabel('Milliseconds')
# plt.show()

df_pit_stops = df_pit_stops[df_pit_stops['milliseconds'] < 500000]

# Converting q1, q2, q3 into milliseconds
def convert_to_seconds(time_str):
    if pd.isnull(time_str):
        return np.nan
    minutes, seconds = time_str.split(':')
    total_seconds = int(minutes) * 60 + float(seconds)
    return total_seconds


columns = ['q1', 'q2', 'q3']

for column in columns:
    df_qualifying[column] = df_qualifying[column].replace('\\N', np.nan)
    df_qualifying[column] = df_qualifying[column].apply(convert_to_seconds)

# Storing the average of q1, q2, q3 in time column
df_qualifying['Qualifying Time'] = df_qualifying[['q1', 'q2', 'q3']].mean(axis=1).round(3)

# Results

df_results['position'] = df_results['position'].replace('\\N', 0)
df_results['milliseconds'] = df_results['milliseconds'].replace('\\N', 0)
df_results['fastestLapTime'] = df_results['fastestLapTime'].replace('\\N', 0)
df_results['fastestLapSpeed'] = df_results['fastestLapSpeed'].replace('\\N', 0)

df_results['position'] = df_results['position'].astype(int)
df_results['milliseconds'] = df_results['milliseconds'].astype(float)
df_results['fastestLapSpeed'] = df_results['fastestLapSpeed'].astype(float)

df_results['milliseconds'] = df_results['milliseconds'] / 1000
df_results = df_results.rename(columns={'milliseconds': 'seconds'})

df_results = df_results.merge(df_races[['raceId', 'date', 'circuitId']], on='raceId')
# Time converted from M:S.ms to ms
# Dates added

# Sprint Results

df_sprint_results['fastestLapTime'] = df_sprint_results['fastestLapTime'].replace('\\N', np.nan)

df_sprint_results = df_sprint_results.merge(df_races[['raceId', 'date']], on='raceId')

# Weather

df_weather.drop(columns='Time', inplace=True)


# Function to get store Round Year-wise and calculate average
def weatherAverage(df_weather, year):
    df_year = df_weather[df_weather['Year'] == year]
    df_weather = df_weather[df_weather['Year'] != year]
    rainfall = df_weather['Rainfall']
    df_year = df_year.groupby('Round Number').mean().round(1).reset_index()

    df_weather = pd.concat([df_weather, df_year])
    return df_weather


years = [x for x in range(2018, 2024)]

for year in years:
    df_weather = weatherAverage(df_weather, year)

# Now I'll add the circuitId and RaceId for to the weather dataset
df_weather = df_weather.rename(columns={"Year": "year", "Round Number": "round"})
df_weather = pd.merge(df_weather, df_races[['raceId', 'circuitId', 'year', 'round']], on=['year', 'round'], how='left')

# Circuits
import folium

# Create a Map instance
m = folium.Map(location=[20, 0], zoom_start=2)

for idx, row in df_circuits.iterrows():
    # Place marker for each circuit
    folium.Marker([row['lat'], row['lng']], popup=f"{row['location']}, {row['country']}").add_to(m)

# Show the map
# print(m)

df_constructor_points = df_constructor_standings.groupby('name')['points'].sum().sort_values(
    ascending=False).reset_index()

df_constructor_wins = df_constructor_standings[df_constructor_standings['position'] == 1].groupby(
    'name').size().reset_index()
df_constructor_wins = df_constructor_wins.rename(columns={0: 'Wins'})

# Plotting Graph
df = pd.merge(df_constructor_points, df_constructor_wins, on='name')

# Create a Figure instance
fig = go.Figure()

# Add a line for each team
for team in df['name'].unique():
    df_team = df[df['name'] == team]
    fig.add_trace(go.Scatter(x=df_team['Wins'], y=df_team['points'], mode='lines+markers', name=team))

# Update layout
fig.update_layout(title_text='<b>Constructor Points vs Wins</b>', titlefont=dict(family='Arial, sans-serif', size=30),
                  title_x=0.5, xaxis_title="Wins", yaxis_title="Points")

# Show the plot
# fig.show()

# Correlation Factor
correlation = df['points'].corr(df['Wins'])
print(f"constructor win correaltion: {correlation}")

df_driver_points = df_driver_standings.groupby('Name')['points'].sum().sort_values(ascending=False).reset_index()

df_driver_wins = df_driver_standings[df_driver_standings['position'] == 1].groupby('Name').size().reset_index()
df_driver_wins = df_driver_wins.rename(columns={0: 'Wins'})

# Plotting Graph
df = pd.merge(df_driver_points, df_driver_wins, on='Name')

# Create a Figure instance
fig = go.Figure()

# Add a line for each team
for team in df['Name'].unique():
    df_team = df[df['Name'] == team]
    fig.add_trace(go.Scatter(x=df_team['Wins'], y=df_team['points'], mode='lines+markers', name=team))

# Update layout
fig.update_layout(title_text='<b>Driver Points vs Wins</b>', titlefont=dict(family='Arial, sans-serif', size=30),
                  title_x=0.5, xaxis_title="Wins", yaxis_title="Points")

# Show the plot
# fig.show()

# Correlation Factor
correlation = df['points'].corr(df['Wins'])
print(f"pilots win correaltion: {correlation}")

df_grid_wins = df_results[['raceId', 'driverId', 'grid', 'position']].copy()
df_grid_wins['Win'] = df_grid_wins['position'].apply(lambda x: 1 if x == 1 else 0)

mean_wins_by_grid = (df_grid_wins.groupby('grid')['Win'].mean() * 100).round(2)
print(mean_wins_by_grid)

threshold = 0.5
percentage_rain = (((df_weather['Rainfall'] > threshold).sum() / len(df_weather)) * 100).round(2)
print("Percent of races when it rained more than 50% of the race: ", percentage_rain)
# 4% des courses ne reprensetents meme pas une course

races_in_rain = df_weather[df_weather['Rainfall'] > 0]

# Before we saw that grid 1, 2, 3 are likely to win 88% of the races. We would like to see how it changes during rain. Note that data is very less.

df_rain_grid_wins = df_grid_wins[df_grid_wins['raceId'].isin(races_in_rain['raceId'])]

mean_wins_by_grid = (df_rain_grid_wins.groupby('grid')['Win'].mean() * 100).round(2)

# Constructor Experience (By races, +1 for each driver)

df_constructor_experience = df_results[['raceId', 'constructorId', 'date']].sort_values('date')

# Calculate the cumulative count of each constructor
df_constructor_experience['Constructor Experience'] = df_constructor_experience.groupby('constructorId').cumcount() + 1

df_driver_experience = df_results[['raceId', 'driverId', 'date']].sort_values('date')

# Calculate the cumulative count of each driver
df_driver_experience['Driver Experience'] = df_driver_experience.groupby('driverId').cumcount() + 1

df_age = df_results[['raceId', 'driverId', 'date']]
df_age = pd.merge(df_age, df_drivers[['driverId', 'dob']], on='driverId', how='left')

# Convert 'dob' and 'date' to datetime if they are not already
df_age['dob'] = pd.to_datetime(df_age['dob'])
df_age['date'] = pd.to_datetime(df_age['date'])

# Calculate age at the time of each race
df_age['Driver Age'] = (df_age['date'] - df_age['dob']).dt.days // 365

# Create the new dataframe with 'raceId', 'driverId', and 'age'
df_driver_age = df_age[['raceId', 'driverId', 'Driver Age']]

df_driver_wins = df_results[['raceId', 'driverId', 'position', 'date']].sort_values('date')

# Create a new column 'Win' which is 1 if the position is 1, else 0
df_driver_wins['Win'] = df_driver_wins['position'].apply(lambda x: 1 if x == 1 else 0)

# Calculate the cumulative sum of wins for each driver
df_driver_wins['Driver Wins'] = df_driver_wins.groupby('driverId')['Win'].cumsum()

df_constructor_wins = df_constructor_standings[['raceId', 'constructorId', 'position', 'date']].sort_values('date')

# Create a new column 'Win' which is 1 if the position is 1, else 0
df_constructor_wins['Win'] = df_constructor_wins['position'].apply(lambda x: 1 if x == 1 else 0)

# Calculate the cumulative sum of wins for each driver
df_constructor_wins['Constructor Wins'] = df_constructor_wins.groupby('constructorId')['Win'].cumsum()

df_constructor_wins['Constructor Wins'] = df_constructor_wins['Constructor Wins'].astype(int)

df_driver_constructor_exp = df_results[['raceId', 'constructorId', 'driverId', 'date']].sort_values('date')

df_driver_constructor_exp['Driver Constructor Experience'] = df_driver_constructor_exp.groupby(
    ['driverId', 'constructorId']).cumcount() + 1

df_status = df_status[~df_status['status'].str.contains("\+\d+ Laps")]
df_status = df_status.drop(0)

df_finish = df_results[['raceId', 'driverId', 'constructorId', 'statusId', 'date']].copy()
df_finish.loc[:, 'Finish'] = (~df_finish['statusId'].isin(df_status['statusId'])).astype(int)
df_finish['date'] = pd.to_datetime(df_finish['date'])

# Sort the DataFrame by date in ascending order
df_finish = df_finish.sort_values('date')

# Calculate the cumulative average of the 'Finish' column for each constructor
df_finish['DNF Score'] = df_finish.groupby('constructorId')['Finish'].expanding().mean().round(2).reset_index(level=0,
                                                                                                              drop=True)

formula_1 = df_results[
    ['raceId', 'driverId', 'constructorId', 'grid', 'position', 'laps', 'seconds', 'fastestLapSpeed', 'date',
     'circuitId']]

# Circuits Length and Turns
formula_1 = formula_1.merge(df_circuits[['circuitId', 'Length', 'Turns']], on='circuitId', how='left')

# Constructor Experience (No. of races by drivers)
formula_1 = formula_1.merge(df_constructor_experience[['raceId', 'constructorId', 'Constructor Experience']],
                            on=['raceId', 'constructorId'], how='left')

# Driver Experience (No. of races)
formula_1 = formula_1.merge(df_driver_experience[['raceId', 'driverId', 'Driver Experience']],
                            on=['raceId', 'driverId'], how='left')

# Driver Age
formula_1 = formula_1.merge(df_driver_age[['raceId', 'driverId', 'Driver Age']], on=['raceId', 'driverId'], how='left')

# Driver Wins
formula_1 = formula_1.merge(df_driver_wins[['raceId', 'driverId', 'Driver Wins']], on=['raceId', 'driverId'],
                            how='left')

# Constructor Wins
formula_1 = formula_1.merge(df_constructor_wins[['raceId', 'constructorId', 'Constructor Wins']],
                            on=['raceId', 'constructorId'], how='left')

# Driver Experience with Constructor
formula_1 = formula_1.merge(
    df_driver_constructor_exp[['raceId', 'constructorId', 'driverId', 'Driver Constructor Experience']],
    on=['raceId', 'constructorId', 'driverId'], how='left')

# DNF Score
formula_1 = formula_1.merge(df_finish[['raceId', 'constructorId', 'DNF Score']], on=['raceId', 'constructorId'],
                            how='left')

formula_1 = formula_1.sort_values(['driverId', 'date'])
formula_1['prev_position'] = formula_1.groupby('driverId')['position'].shift(1)
formula_1['prev_position'] = formula_1['prev_position'].fillna(0)

formula_1 = formula_1.drop_duplicates(subset=['raceId', 'driverId', 'constructorId'], keep='last')

# Change second argument to select positions for prediction
pos = list(range(1, 21))
formula_1 = formula_1[formula_1['position'].isin(pos)]

formula_1 = formula_1[formula_1['Constructor Wins'].notnull()]

formula_1 = formula_1[formula_1['date'] >= '1980-01-01']

# formula_1['podium'] = formula_1['position'].apply(lambda x: x) # All Positions
# formula_1['podium'] = formula_1['position'].apply(lambda x: x if 1<=x<=3 else 0) # All Positions, (this creates a false boost in accuracy)
formula_1['podium'] = formula_1['position'].apply(lambda x: x if 1 <= x <= 3 else 0)

formula_1.to_csv('formula1.csv', index=False)

from sklearn.model_selection import train_test_split

# Drop 'position' and other unnecessary columns
X = formula_1.drop(['position', 'seconds', 'podium', 'date', 'fastestLapSpeed', 'raceId'], axis=1)
y = formula_1['podium']  # target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Testing different models with Cross Validation Score

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Initialize the models
clf1 = RandomForestClassifier(random_state=42)
clf2 = SVC(random_state=42)
clf3 = KNeighborsClassifier()

# List of models
models = [clf1, clf2, clf3]

# Dictionary to hold the model names and their scores
scores = {}

for model in models:
    model_name = model.__class__.__name__
    score = cross_val_score(model, X, y, cv=5).mean()
    scores[model_name] = score

# Print the scores
for model, score in scores.items():
    print(f"{model}: {score:.2f}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Define the pipeline
formula1_predict = Pipeline([
    ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
    # Use RandomForestClassifier with n_estimators=50
])

# Fit the pipeline on the training data
formula1_predict.fit(X, y)

# Predict the target variable for the test set
y_pred = formula1_predict.predict(X_test)

# Accuracy and Cross Validation Scores
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

rfc_accuracy = accuracy_score(y_test, y_pred)
rfc_cv = cross_val_score(formula1_predict, X, y, cv=5)

print(f"Accuracy of the Random Forest Classifier: {rfc_accuracy * 100:.2f}%")
print(f"Average cross-validation score: {rfc_cv.mean() * 100:.2f}%")

from joblib import dump

dump(formula1_predict, 'formula1_model.joblib')


def prediction(driver_name, grid, circuit_loc):
    driver = df_drivers.loc[df_drivers['Name'] == driver_name, 'driverId'].iloc[0]
    circuit = df_circuits.loc[df_circuits['location'] == circuit_loc, ['circuitId', 'laps']].iloc[0]

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
    # print(features)

    return formula1_predict.predict(features), formula1_predict.predict_proba(features)

# Drivers is the list of driver in the race, sorted by their grid position
drivers = ['Max Verstappen', 'Charles Leclerc', 'George Russell', 'Carlos Sainz', 'Sergio Pérez', 'Fernando Alonso', 'Lando Norris', 'Oscar Piastri', 'Lewis Hamilton', 'Nico Hülkenberg', 'Yuki Tsunoda', 'Lance Stroll', 'Alexander Albon', 'Daniel Ricciardo', 'Kevin Magnussen', 'Valtteri Bottas', 'Logan Sargeant', 'Esteban Ocon', 'Pierre Gasly']

# Grids is a list of grid positions from your table
grids = list(range(1, 21))

# Location of circuit
circuit_loc = 'Sakhir'

predictions = []

# Iterate over drivers and their corresponding grid positions
for driver_name, grid in zip(drivers, grids):
    # Call your prediction function and print the result
    pred, prob = prediction(driver_name, grid, circuit_loc)
    if pred in [1, 2, 3]:
        predictions.append({
        'Driver Name ': driver_name,
        'Grid': grid,
        'Prediction': pred,
        'Probability': np.max(prob)
        })
    # print(f'{driver_name}, {grid}, {pred}, prob: {prob}')

# print(predictions)