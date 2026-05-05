import pandas as pd
import numpy as np


def compute_constructor_experience(df_results):
    """Compute cumulative race count per constructor as a proxy for team experience."""
    df = df_results[['raceId', 'constructorId', 'date']].sort_values('date')
    df['Constructor Experience'] = df.groupby('constructorId').cumcount() + 1
    return df


def compute_driver_experience(df_results):
    """Compute cumulative race count per driver as a proxy for driver experience."""
    df = df_results[['raceId', 'driverId', 'date']].sort_values('date')
    df['Driver Experience'] = df.groupby('driverId').cumcount() + 1
    return df


def compute_driver_age(df_results, df_drivers):
    """Compute each driver's age in whole years at the time of each race."""
    df = df_results[['raceId', 'driverId', 'date']]
    df = pd.merge(df, df_drivers[['driverId', 'dob']], on='driverId', how='left')

    df['dob'] = pd.to_datetime(df['dob'])
    df['date'] = pd.to_datetime(df['date'])

    # Integer division of days gives age in full years
    df['Driver Age'] = (df['date'] - df['dob']).dt.days // 365
    return df[['raceId', 'driverId', 'Driver Age']]


def compute_driver_wins(df_results):
    """Compute cumulative win count per driver up to and including each race."""
    df = df_results[['raceId', 'driverId', 'position', 'date']].sort_values('date')
    df['Win'] = df['position'].apply(lambda x: 1 if x == 1 else 0)
    df['Driver Wins'] = df.groupby('driverId')['Win'].cumsum()
    return df


def compute_constructor_wins(df_constructor_standings):
    """Compute cumulative win count per constructor up to and including each race."""
    df = df_constructor_standings[['raceId', 'constructorId', 'position', 'date']].sort_values('date')
    df['Win'] = df['position'].apply(lambda x: 1 if x == 1 else 0)
    df['Constructor Wins'] = df.groupby('constructorId')['Win'].cumsum()
    df['Constructor Wins'] = df['Constructor Wins'].astype(int)
    return df


def compute_driver_constructor_experience(df_results):
    """Compute cumulative races each driver has completed with their current constructor."""
    df = df_results[['raceId', 'constructorId', 'driverId', 'date']].sort_values('date')
    df['Driver Constructor Experience'] = df.groupby(['driverId', 'constructorId']).cumcount() + 1
    return df


def compute_dnf_score(df_results, df_status):
    """Compute rolling finish rate per constructor as a DNF reliability score."""
    # Remove "+N Laps" entries — those are classified finishes, not DNFs
    status_dnf = df_status[~df_status['status'].str.contains(r"\+\d+ Laps")]
    # Drop index 0 ("Finished") so it is not counted as a bad status
    status_dnf = status_dnf.drop(0)

    df_finish = df_results[['raceId', 'driverId', 'constructorId', 'statusId', 'date']].copy()
    # Finish=1 if the statusId is NOT in the non-finish (DNF) list
    df_finish.loc[:, 'Finish'] = (~df_finish['statusId'].isin(status_dnf['statusId'])).astype(int)
    df_finish['date'] = pd.to_datetime(df_finish['date'])
    df_finish = df_finish.sort_values('date')

    # Expanding mean gives a running average finish rate up to each race
    df_finish['DNF Score'] = (
        df_finish.groupby('constructorId')['Finish']
        .expanding().mean().round(2)
        .reset_index(level=0, drop=True)
    )
    return df_finish


def build_formula1(df_results, df_circuits, df_constructor_experience, df_driver_experience,
                   df_driver_age, df_driver_wins, df_constructor_wins,
                   df_driver_constructor_exp, df_finish):
    """Merge all engineered features into the modelling dataframe and save to CSV."""
    formula_1 = df_results[[
        'raceId', 'driverId', 'constructorId', 'grid', 'position',
        'laps', 'seconds', 'fastestLapSpeed', 'date', 'circuitId'
    ]]

    # Circuit characteristics
    formula_1 = formula_1.merge(df_circuits[['circuitId', 'Length', 'Turns']], on='circuitId', how='left')

    # Constructor Experience (total races entered by the team up to this point)
    formula_1 = formula_1.merge(
        df_constructor_experience[['raceId', 'constructorId', 'Constructor Experience']],
        on=['raceId', 'constructorId'], how='left'
    )

    # Driver Experience (total races entered by the driver)
    formula_1 = formula_1.merge(
        df_driver_experience[['raceId', 'driverId', 'Driver Experience']],
        on=['raceId', 'driverId'], how='left'
    )

    formula_1 = formula_1.merge(
        df_driver_age[['raceId', 'driverId', 'Driver Age']],
        on=['raceId', 'driverId'], how='left'
    )

    formula_1 = formula_1.merge(
        df_driver_wins[['raceId', 'driverId', 'Driver Wins']],
        on=['raceId', 'driverId'], how='left'
    )

    formula_1 = formula_1.merge(
        df_constructor_wins[['raceId', 'constructorId', 'Constructor Wins']],
        on=['raceId', 'constructorId'], how='left'
    )

    # Driver experience specifically with their current constructor
    formula_1 = formula_1.merge(
        df_driver_constructor_exp[['raceId', 'constructorId', 'driverId', 'Driver Constructor Experience']],
        on=['raceId', 'constructorId', 'driverId'], how='left'
    )

    formula_1 = formula_1.merge(
        df_finish[['raceId', 'constructorId', 'DNF Score']],
        on=['raceId', 'constructorId'], how='left'
    )

    # Lagged position: driver's finishing position in their previous race
    formula_1 = formula_1.sort_values(['driverId', 'date'])
    formula_1['prev_position'] = formula_1.groupby('driverId')['position'].shift(1)
    formula_1['prev_position'] = formula_1['prev_position'].fillna(0)

    formula_1 = formula_1.drop_duplicates(subset=['raceId', 'driverId', 'constructorId'], keep='last')

    # Restrict to classified finishing positions (1–20)
    pos = list(range(1, 21))
    formula_1 = formula_1[formula_1['position'].isin(pos)]

    formula_1 = formula_1[formula_1['Constructor Wins'].notnull()]

    # Use data from 2022 onward (post-regulation change era)
    formula_1 = formula_1[formula_1['date'] >= '2022-01-01']

    # Target: podium position (1/2/3) or 0 for outside top 3
    formula_1['podium'] = formula_1['position'].apply(lambda x: x if 1 <= x <= 3 else 0)

    formula_1.to_csv('data/formula1.csv', index=False)
    return formula_1
