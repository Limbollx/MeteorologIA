import numpy as np
import pandas as pd
import os

# Définition des chemins et du fichier de données

dirPath = os.path.dirname(os.path.realpath(__file__))  # Chemin du script

dirSrc = dirPath[0:dirPath.rfind(os.sep)]  # Répertoire parent
adr = dirSrc + os.sep + 'data' + os.sep + 'meteo_st-pierre_2024.csv'

# Constantes pour les calculs thermiques
T_corps = 37  # Température corporelle (°C)
R_th = 0.25  # Résistance thermique (m²K/W)
M_sueur = 1.3  # Masse de sueur évaporée (kg/jour)
S_corps = 1.5  # Surface corporelle (m²)

# Chargement et filtrage des données à 12h
df = pd.read_csv(adr, sep=";", parse_dates=["date"])
df_filtered = df[df['date'].dt.hour == 12].reset_index(drop=True)

def get_index(day, time):
    target_datetime = pd.Timestamp(f"{day} {time}", tz='+04:00')
    
    # Trouver l'index de la ligne correspondant à la date cible
    target_index = df[df['date'] == target_datetime].index
    return target_index[0]

def reading(day, time):
    """Calcule la température ressentie en fonction des paramètres météorologiques."""

    index = get_index(day, time)

    RH = df.at[index, 'RH'] / 100
    T_station = df.at[index, 'Tair']
    vent_station = df.at[index, 'Ws10'] * 3.6  # Conversion en km/h
    Qvap = 2257000  # Chaleur latente d'évaporation de l'eau (J/kg)

    h = (5 + 7.2 * np.sqrt(vent_station)) if vent_station >= 20 else (8 + 10 * np.sqrt(vent_station))
    
    # Calcul des flux thermiques
    Phi_temp = (T_corps - T_station) / R_th
    Phi_solaire = 300  # Hypothèse constante
    Phi_vent = -h * (T_corps - T_station)
    Phi_rh = -((M_sueur * Qvap) / (S_corps * (60**2) * 24)) * (1 - RH)
    Phi_rh_standard = -((M_sueur * Qvap) / (S_corps * (60**2) * 24)) * (1 - 0.5)
    Phi_corps = 100 / S_corps
    
    Phi_total = sum([Phi_solaire, Phi_temp, Phi_vent, Phi_rh, Phi_corps])
    T_ressentie = (Phi_total - Phi_rh - Phi_corps + Phi_rh_standard) * R_th + T_corps
    
    print(f'Temp ressentie: {T_ressentie:.2f}°C le {str(df["date"][index])[:-15]} à {str(df["date"][index])[11:-9]}')
    
    return Phi_total, T_ressentie, Phi_solaire, Phi_temp, Phi_vent, Phi_rh, Phi_corps, h, vent_station, RH

# Possibilité de vérifier une date donnée
# reading(0)
reading('2024-01-01', '18:00')