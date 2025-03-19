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

def reading(n):
    """Calcule la température ressentie en fonction des paramètres météorologiques."""
    RH = df_filtered.at[n, 'RH'] / 100
    T_station = df_filtered.at[n, 'Tair']
    vent_station = df_filtered.at[n, 'Ws10'] * 3.6  # Conversion en km/h
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
    
    print(f'Temp ressentie: {T_ressentie.round(1)}°C le {str(df_filtered["date"][n])[:-15]} à {str(df_filtered["date"][n])[11:-9]}')
    
    return Phi_total, T_ressentie, Phi_solaire, Phi_temp, Phi_vent, Phi_rh, Phi_corps, h, vent_station, RH

def twodimtable():
    """Génère une table 2D de températures ressenties en fonction de l'humidité et du vent."""
    rdf = pd.DataFrame()
    RH_values, wind_speed_values, temp_values = [], [], {}
    
    for n in range(len(df_filtered)):
        Phi_total, T_ressentie, Phi_solaire, Phi_temp, Phi_vent, Phi_rh, Phi_corps, h, vent_station, RH = reading(n)
        wind_speed = 5 * round(vent_station / 5)
        RH_percent = 5 * round((RH * 100) / 5)
        
        RH_values.append(RH_percent)
        wind_speed_values.append(wind_speed)
        temp_values.setdefault((RH_percent, wind_speed), []).append(T_ressentie)
    
    unique_RH = sorted(set(RH_values))
    unique_wind_speeds = sorted(set(wind_speed_values))
    rdf = pd.DataFrame(index=unique_RH, columns=unique_wind_speeds)
    
    for (rh, wind_speed), temps in temp_values.items():
        rdf.at[rh, wind_speed] = np.mean(temps).round(0)
    
    return rdf

def processing(n):
    """Met à jour le DataFrame des résultats avec les flux thermiques et la température ressentie."""
    Phi_total, T_ressentie, *_ = reading(n)
    results.at[results.index[n], 'flux total'] = Phi_total
    results.at[results.index[n], 'temperature ressentie'] = T_ressentie

def verifying(n):
    """Vérifie les valeurs thermiques pour une date donnée."""
    date_ref = pd.Timestamp("2024-01-01")
    date_target = pd.Timestamp(n)
    u = (date_target - date_ref).days
    
    Phi_total, T_ressentie, Phi_solaire, Phi_temp, Phi_vent, Phi_rh, Phi_corps, h, vent_station = reading(u)
    
    values = [Phi_solaire, Phi_temp, Phi_vent, Phi_rh, Phi_corps, h, vent_station, T_ressentie]
    labels = ["Phi_solaire", "Phi_temp", "Phi_vent", "Phi_rh", "Phi_corps", "h", "vent_station", "T_ressentie"]
    
    for label, value in zip(labels, values):
        print(label, ": ", value)

# Création du DataFrame des résultats
results = pd.DataFrame(index=df_filtered['date'])
results.index = results.index.strftime('%Y-%m-%d')

# Exécution de la fonction de lecture sur le premier point de données
reading(1)

# Possibilité de vérifier une date donnée
# verifying('2024-01-02')

# Boucle de traitement sur l'ensemble des données (commentée pour éviter une exécution directe)
# for i in range(len(df_filtered)):
#     processing(i)

# Affichage des résultats (commenté par défaut)
# print(results)