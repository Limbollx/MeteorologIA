import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import OtherProcessing as op

# Définition des chemins et du fichier de données
dirPath = os.path.dirname(os.path.realpath(__file__))  # Chemin du script

dirSrc = dirPath[0:dirPath.rfind(os.sep)]  # Répertoire parent
adr = dirSrc + os.sep + 'data' + os.sep + 'full-data-st_pierre-2024.csv'

# Constantes pour les calculs thermiques
T_corps = 37  # Température corporelle (°C)
R_th = 0.15  # Résistance thermique (m²K/W)
M_sueur = 1.3  # Masse de sueur évaporée (kg/jour)
S_corps = 1.5  # Surface corporelle (m²)
T_peau = 30 # Température de la peau (C°)
t_s = 6.5
t_c = 18.5
heures = np.linspace(0, 24, 1000)

I_values = np.array([op.solar_irradiance(t, t_s, t_c) for t in heures])

am = I_values[:np.argmax(I_values)]
pm = I_values[np.argmax(I_values):]

# Chargement des données
df = pd.read_csv(adr, sep=";", parse_dates=["date"])

def get_index(daytime):
    target_datetime = pd.Timestamp(daytime, tz='+04:00')
    
    # Trouver l'index de la ligne correspondant à la date cible
    target_index = df[df['date'] == target_datetime].index
    return target_index[0]

def reading(daytime, index):
    """Calcule la température ressentie en fonction des paramètres météorologiques."""

    # index = get_index(daytime)

    RH = df.at[index, 'RH'] / 100
    T_station = df.at[index, 'Tair']
    vent_station = df.at[index, 'Ws10'] * 3.6  # Conversion en km/h
    # Qvap = 2257000  # Chaleur latente d'évaporation de l'eau (J/kg)

    # h = (5 + 7.2 * np.sqrt(vent_station)) if vent_station >= 20 else (8 + 10 * np.sqrt(vent_station))
    h = op.fonction_logistique((5 + 7.2 * np.sqrt(vent_station)),
                            (8 + 10 * np.sqrt(vent_station)),
                             vent_station)

    # Calcul des flux thermiques
    Phi_temp = ((T_corps - T_station) / R_th) * S_corps

    time = op.time_to_float(daytime)
    if time <= 12.5:
        dirSol = op.fonction_logistique(S_corps/2, np.pi*0.0276, time ,change=9)
    if time >= 12.5:
        dirSol = op.fonction_logistique(S_corps/2, np.pi*0.0276, time ,change=16.5)
    Phi_solaire = df.at[index, 'Rglo'] * dirSol

    Phi_vent = -h * (T_corps - T_station) * S_corps

    Phi_rh = (0.01*610.94*(np.exp((17.625*T_peau)/
                    (T_peau+243.04))-RH*np.exp((17.625*T_station)/
                    (T_station+243.04)))) * S_corps
    
    Phi_rh_standard = (0.01*610.94*(np.exp((17.625*T_peau)/
                    (T_peau+243.04))-0.5*np.exp((17.625*T_station)/
                    (T_station+243.04)))) * S_corps
    
    Phi_corps = 100
    
    Phi_total = sum([Phi_solaire, Phi_temp, Phi_vent, Phi_rh, Phi_corps])
    T_ressentie = (Phi_total - Phi_corps - Phi_rh_standard) * R_th + T_corps
    
    # print(f'Temp ressentie: {T_ressentie:.2f}°C le {str(df["date"][index])[:-15]} à {str(df["date"][index])[11:-9]}')
    
    return T_ressentie



JourActuel = '2024-01-01-12:00'

# reading('2024-03-27-12:00')

# while JourActuel != '2025-01-01-00:00':

resultat = np.zeros(8773)
heure = np.arange(8773)
while JourActuel != '2025-01-01-00:00':
# for _ in range(100):
    index = get_index(JourActuel)
    save = reading(JourActuel, index)

    resultat[index] = float(f'{save:.2f}')
    JourActuel = op.augmente_heure(JourActuel)

plt.scatter(heure, resultat)
plt.axhline(0)
plt.show()