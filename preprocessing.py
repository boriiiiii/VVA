import pandas as pd
import numpy as np


def preprocess_circuits(df_circuits):
    """Clean altitude, impute missing Length/Turns with column means, compute lap count, and save."""
    # Replace '\N' values with NaN
    df_circuits['alt'] = df_circuits['alt'].replace('\\N', np.nan)
    df_circuits['alt'] = df_circuits['alt'].astype(float)

    # Replace 0s with column mean to impute scrape failures
    df_circuits['Length'] = df_circuits['Length'].replace(
        0, df_circuits[df_circuits['Length'] != 0]['Length'].mean()
    ).astype(float)
    df_circuits['Turns'] = df_circuits['Turns'].replace(
        0, df_circuits[df_circuits['Turns'] != 0]['Turns'].mean()
    ).astype(int)
    df_circuits.loc[:, 'alt'] = df_circuits['alt'].fillna(df_circuits['alt'].mean()).astype(int)

    # Estimate lap count assuming a 305 km race distance
    df_circuits['laps'] = (305 / df_circuits['Length']).astype(int)

    df_circuits.to_csv('data/circuits.csv', index=False)
    return df_circuits


def _replace_code(row):
    """Return existing code or derive a 3-letter code from the cleaned surname."""
    if row['code'] == '\\N':
        return row['surname'][:3].upper()
    else:
        return row['code']


def preprocess_drivers(df_drivers):
    """Drop unused columns, build Name field, fix missing codes, and save."""
    # No need for driver number
    df_drivers.drop('number', axis=1, inplace=True)

    # Add full Name column instead of separate forename/surname
    df_drivers['Name'] = df_drivers['forename'] + ' ' + df_drivers['surname']

    # Clean surname before using it to generate missing codes
    df_drivers['surname'] = df_drivers['surname'].str.replace(' ', '')
    df_drivers['code'] = df_drivers.apply(_replace_code, axis=1)

    df_drivers.drop(columns=['driverRef', 'forename', 'surname', 'url'], axis=1, inplace=True)

    df_drivers.to_csv('data/drivers.csv', index=False)
    return df_drivers


def preprocess_standings(df_constructor_standings, df_constructors, df_races):
    """Merge constructor name and race date into constructor standings."""
    df_constructor_standings = df_constructor_standings.merge(
        df_constructors[['constructorId', 'name']], on='constructorId', how='left'
    )
    df_constructor_standings = df_constructor_standings.merge(
        df_races[['raceId', 'date']], on='raceId'
    )
    return df_constructor_standings


def preprocess_driver_standings(df_driver_standings, df_drivers):
    """Merge driver Name into driver standings."""
    df_driver_standings = df_driver_standings.merge(
        df_drivers[['driverId', 'Name']], on='driverId', how='left'
    )
    return df_driver_standings


def preprocess_lap_times(df_lap_times, df_races):
    """Filter lap times to known races, flag IQR outliers, and remove extreme values."""
    df_lap_times = df_lap_times[df_lap_times['raceId'].isin(df_races['raceId'])]

    Q1 = df_lap_times['milliseconds'].quantile(0.25)
    Q3 = df_lap_times['milliseconds'].quantile(0.75)
    IQR = Q3 - Q1

    # IQR-based outlier flag (does not filter rows — kept for downstream inspection)
    lap_time_filter = (
        (df_lap_times['milliseconds'] >= Q1 - 1.5 * IQR) &
        (df_lap_times['milliseconds'] <= Q3 + 1.5 * IQR)
    )
    df_lap_times['outlier'] = ~lap_time_filter

    # Hard cap to exclude extremely long laps (e.g. red flag, safety car formation lap)
    df_lap_times = df_lap_times[df_lap_times['milliseconds'] < 600000]
    return df_lap_times


def preprocess_pit_stops(df_pit_stops, df_races):
    """Filter pit stops to known races, flag IQR outliers, and remove extreme values."""
    df_pit_stops = df_pit_stops[df_pit_stops['raceId'].isin(df_races['raceId'])]

    Q1 = df_pit_stops['milliseconds'].quantile(0.25)
    Q3 = df_pit_stops['milliseconds'].quantile(0.75)
    IQR = Q3 - Q1

    # IQR-based outlier flag
    pit_filter = (
        (df_pit_stops['milliseconds'] >= Q1 - 1.5 * IQR) &
        (df_pit_stops['milliseconds'] <= Q3 + 1.5 * IQR)
    )
    df_pit_stops['outlier'] = ~pit_filter

    # Hard cap to exclude exceptionally long pit stops
    df_pit_stops = df_pit_stops[df_pit_stops['milliseconds'] < 500000]
    return df_pit_stops


def _convert_to_seconds(time_str):
    """Convert a M:S.ms qualifying time string to total seconds as a float."""
    if pd.isnull(time_str):
        return np.nan
    minutes, seconds = time_str.split(':')
    total_seconds = int(minutes) * 60 + float(seconds)
    return total_seconds


def preprocess_qualifying(df_qualifying):
    """Convert Q1/Q2/Q3 lap times to seconds and store their mean as Qualifying Time."""
    columns = ['q1', 'q2', 'q3']
    for column in columns:
        df_qualifying[column] = df_qualifying[column].replace('\\N', np.nan)
        df_qualifying[column] = df_qualifying[column].apply(_convert_to_seconds)

    # Average across available qualifying sessions
    df_qualifying['Qualifying Time'] = df_qualifying[['q1', 'q2', 'q3']].mean(axis=1).round(3)
    return df_qualifying


def preprocess_results(df_results, df_races):
    """Clean result columns, convert milliseconds to seconds, and merge race metadata."""
    df_results['position'] = df_results['position'].replace('\\N', 0)
    df_results['milliseconds'] = df_results['milliseconds'].replace('\\N', 0)
    df_results['fastestLapTime'] = df_results['fastestLapTime'].replace('\\N', 0)
    df_results['fastestLapSpeed'] = df_results['fastestLapSpeed'].replace('\\N', 0)

    df_results['position'] = df_results['position'].astype(int)
    df_results['milliseconds'] = df_results['milliseconds'].astype(float)
    df_results['fastestLapSpeed'] = df_results['fastestLapSpeed'].astype(float)

    # Convert race duration from milliseconds to seconds
    df_results['milliseconds'] = df_results['milliseconds'] / 1000
    df_results = df_results.rename(columns={'milliseconds': 'seconds'})

    # Add race date and circuit for time-series feature engineering
    df_results = df_results.merge(df_races[['raceId', 'date', 'circuitId']], on='raceId')
    return df_results


def preprocess_sprint_results(df_sprint_results, df_races):
    """Clean sprint result lap times and merge race date."""
    df_sprint_results['fastestLapTime'] = df_sprint_results['fastestLapTime'].replace('\\N', np.nan)
    df_sprint_results = df_sprint_results.merge(df_races[['raceId', 'date']], on='raceId')
    return df_sprint_results


def _weather_average(df_weather, year):
    """Average weather readings per round within a year, then reinsert into df_weather."""
    df_year = df_weather[df_weather['Year'] == year]
    df_weather = df_weather[df_weather['Year'] != year]
    df_year = df_year.groupby('Round Number').mean().round(1).reset_index()
    df_weather = pd.concat([df_weather, df_year])
    return df_weather


def preprocess_weather(df_weather, df_races):
    """Aggregate weather by round, then attach raceId and circuitId."""
    df_weather.drop(columns='Time', inplace=True)

    years = list(range(2018, 2024))
    for year in years:
        df_weather = _weather_average(df_weather, year)

    # Rename for merge compatibility with races table
    df_weather = df_weather.rename(columns={"Year": "year", "Round Number": "round"})
    df_weather = pd.merge(
        df_weather,
        df_races[['raceId', 'circuitId', 'year', 'round']],
        on=['year', 'round'],
        how='left'
    )
    return df_weather
