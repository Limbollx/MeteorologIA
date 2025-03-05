import numpy as np
import pandas as pd
import os

# Définition des chemins en cours 
dirPath = os.path.dirname(os.path.realpath(__file__))
dirSrc = dirPath[0:dirPath.rfind(os.sep)]

# Définition des chemins des dossiers et fichiers 
adr = dirSrc + os.sep + 'data' + os.sep + 'meteo_st-pierre_2024.csv'

T_corps = 37
R_th = 0.25
M_sueur = 1.3
S_corps = 1.5

df = pd.read_csv(adr, sep=";", parse_dates=["date"])
df_filtered = df[df['date'].dt.hour == 12].reset_index(drop=True)

def reading(n):
    # RH = df_filtered.at[n, 'RH'] / 100
    # T_station = df_filtered.at[n, 'Tair']
    # vent_station = df_filtered.at[n, 'Ws10'] * 3.6
    RH = 0.84
    T_station =  29
    vent_station = 30
    Qvap = -2446.43 * T_station + 2501875.143
    if vent_station >= 20:
        h = 5 + 7.2 * np.sqrt(vent_station)
    else:
        h = 8 + 10 * np.sqrt(vent_station)

    Phi_temp = (T_corps - T_station) / R_th
    Phi_solaire = 800
    Phi_vent = -h * (T_corps - T_station)
    Phi_rh = -((M_sueur * Qvap) / (S_corps * (60**2) * 24)) * (1 - RH)
    Phi_corps = 100 / S_corps
    Phi_values = [Phi_solaire, Phi_temp, Phi_vent, Phi_rh, Phi_corps]
    Phi_total = sum(Phi_values)

    T_ressentie = (Phi_total - Phi_rh - Phi_corps) * R_th + T_corps

    print('Temp ressentie:', T_ressentie)

    return Phi_total, T_ressentie, Phi_solaire, Phi_temp, Phi_vent, Phi_rh, Phi_corps, h, vent_station, RH


def twodimtable():
    rdf = pd.DataFrame()
    
    RH_values = []
    wind_speed_values = []
    temp_values = {}
    
    for n in range(len(df_filtered)):
        Phi_total, T_ressentie, Phi_solaire, Phi_temp, Phi_vent, Phi_rh, Phi_corps, h, vent_station, RH = reading(n)
        
        wind_speed = 5 * round(vent_station / 5)
        RH_percent = 5 * round((RH * 100) / 5)
        
        RH_values.append(RH_percent)
        wind_speed_values.append(wind_speed)
        
        if (RH_percent, wind_speed) in temp_values:
            temp_values[(RH_percent, wind_speed)].append(T_ressentie)
        else:
            temp_values[(RH_percent, wind_speed)] = [T_ressentie]
    
    unique_RH = sorted(set(RH_values))
    unique_wind_speeds = sorted(set(wind_speed_values))
    rdf = pd.DataFrame(index=unique_RH, columns=unique_wind_speeds)
    
    for (rh, wind_speed), temps in temp_values.items():
        rdf.at[rh, wind_speed] = np.mean(temps).round(0)
    
    return rdf

def processing(n):
    Phi_total, T_ressentie, *_ = reading(n)
    results.at[results.index[n], 'flux total'] = Phi_total
    results.at[results.index[n], 'temperature ressentie'] = T_ressentie


def verifying(n):
    date_ref = pd.Timestamp("2024-01-01")
    date_target = pd.Timestamp(n)
    u = (date_target - date_ref).days

    Phi_total, T_ressentie, Phi_solaire, Phi_temp, Phi_vent, Phi_rh, Phi_corps, h, vent_station = reading(u)
    values = [Phi_solaire, Phi_temp, Phi_vent, Phi_rh, Phi_corps, h, vent_station, T_ressentie]
    labels = ["Phi_solaire", "Phi_temp", "Phi_vent", "Phi_rh", "Phi_corps", "h", "vent_station", "T_ressentie"]
    
    for label, value in zip(labels, values):
        print(label, ": ", value)


results = pd.DataFrame(index=df_filtered['date'])
results.index = results.index.strftime('%Y-%m-%d')

# print(twodimtable())

reading(0)

# verifying('2024-01-01')

# for i in range(len(df_filtered)):
#     processing(i)

# print(results)