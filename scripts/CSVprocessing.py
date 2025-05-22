'''
 # @ Auteur: Mathéo Guillot--Eid
 # @ Crée le: 2025-01-29 09:11:30
 # @ Modifié par: Mathéo Guillot--Eid
 # @ Modifié le: 2025-05-22 13:08:03
 # @ Description: Script principal pour traiter différente données et calculer la température ressentie
 '''

#--------------------------------------------------
# Importation des librairies
#--------------------------------------------------

import numpy as np
import os
try:
    import pandas as pd
except Exception:
    os.system("pip install pandas")
    import pandas as pd
try:
    import matplotlib.pyplot as plt
except Exception:
    os.system("pip install matplotlib")
    import matplotlib.pyplot as plt

from os.path import dirname, realpath, sep
from sys import path

dirPath = dirname(realpath(__file__))
dirSrc = dirPath[0:dirPath.rfind(sep)]
# Détermine les chemins absolus pour accéder aux ressources du projet
path += [dirSrc, dirPath]

import OtherProcessing as op

#--------------------------------------------------
# Importation des données
#--------------------------------------------------

# Définition des chemins et du fichier de données
dirPath = os.path.dirname(os.path.realpath(__file__))  # Chemin du script

dirSrc = dirPath[0:dirPath.rfind(os.sep)]  # Répertoire parent
adr = dirSrc + os.sep + 'data' + os.sep + 'full-data-st_pierre2-2024.csv'
adrTr = dirSrc + os.sep + 'data' + os.sep + 'data-Tressentie.csv'

# Chargement des données
df_base = pd.read_csv(adr, sep=";", parse_dates=["date"])

#--------------------------------------------------
# Définition des constantes
#--------------------------------------------------

T_corps = 37  # Température corporelle (°C)
R_th = 0.3  # Résistance thermique (m²K/W)
R_th_standard = 0.1  # Résistance thermique standard (m²K/W)
M_sueur = 1.3  # Masse de sueur évaporée (kg/jour)
S_corps = 1.5  # Surface corporelle (m²)
T_peau = 30 # Température de la peau (C°)

t_s = 6.5
t_c = 18.5

heures = np.linspace(0, 24, 1000)

I_values = np.array([op.solar_irradiance(t, t_s, t_c) for t in heures])

am = I_values[:np.argmax(I_values)]
pm = I_values[np.argmax(I_values):]

#--------------------------------------------------
# Définition des fonctions
#--------------------------------------------------

def get_index(df, daytime):
    target_datetime = pd.Timestamp(daytime, tz='+04:00')
    # Trouver l'index de la ligne correspondant à la date cible
    target_index = df[df['date'] == target_datetime].index
    if len(target_index) == 0:
        return None
    return target_index[0]

rm_dt = get_index(df_base, "2024-01-28 21:00+04:00")
df_base = df_base.drop(index=range(rm_dt, rm_dt + 15)).reset_index(drop=True)


def reading(df, daytime, index, show=False):
    """Calcule la température ressentie en fonction des paramètres météorologiques."""

    RH = df.at[index, 'RH'] / 100  # Humidité relative (fraction)
    T_station = df.at[index, 'Tair']  # Température de l'air (°C)
    vent_station = df.at[index, 'Ws10']  # Vitesse du vent (km/h)

    # Coefficient d'échange thermique par convection (W/(m²·K))
    h = op.fonction_logistique(
        8.3 * (vent_station**0.6),
        10.45 - vent_station + (10*np.sqrt(vent_station)),
        vent_station,
        change=10,
        vitesse=1.5
    )


    # Flux thermique entre le corps et l’air (W)
    Phi_temp = ((T_corps - T_station) / R_th)/ S_corps  # (°C / (m²·K/W)) = W/m²

    time = op.time_to_float(daytime)
    # dirSol = np.cos(np.pi/2 * (time - 6) / (18 - 6)) * S_corps/2
    
    if time <= 12.5:
        # Facteur de projection solaire (sans unité)
        dirSol = op.fonction_logistique(S_corps*0.25, np.pi*0.055, time ,change=9, vitesse=2)
    if time > 12.5:
        dirSol = op.fonction_logistique(np.pi*0.055, S_corps*0.25, time ,change=16.5, vitesse=2)

    # Flux solaire incident sur le corps (W)
    Phi_solaire = df.at[index, 'Rglo'] * (dirSol)  # (W/m²) = W/m²

    # Flux thermique dû au vent (W)
    Phi_vent = -h * (T_corps - T_station)  # (W/m²·K) * (°C) = W/m²

    # Flux d’évaporation lié à l’humidité réelle (W)
    Phi_rh = (0.01*610.94*(np.exp((17.625*T_peau)/(T_peau+243.04)) -
              RH*np.exp((17.625*T_station)/(T_station+243.04)))) # Pa ≈ W/m²

    # Flux d’évaporation à humidité relative standard 50% (W)
    Phi_rh_standard = (0.01*610.94*(np.exp((17.625*T_peau)/(T_peau+243.04)) -
                         0.5*np.exp((17.625*T_station)/(T_station+243.04))))  # Pa ≈ W/m²

    Phi_corps = 100 / S_corps  # Flux métabolique corporel de base (W/m²), valeur fixe

    # Somme totale des flux thermiques (W/m²)
    Phi_total = sum([Phi_solaire, Phi_temp, Phi_vent, Phi_rh, Phi_corps])

    # Température ressentie (°C)
    T_ressentie = ((Phi_total - Phi_corps - Phi_rh_standard) * R_th_standard + T_corps)  # (W/m²) * (m²·°C/W) + °C = °C

    if show:
        print(f'Temp ressentie: {T_ressentie:.2f}°C le {str(df["date"][index])[:-15]} à {str(df["date"][index])[11:-9]}')
    
    return T_ressentie

def draw(df=df_base, JourActuel='2024-01-01-12:00', drawing=True):
    # Convertir la colonne 'date' en datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Initialiser un dictionnaire pour stocker les températures par jour
    daily_temps = {}
    dates = []
    means = []
    
    last_date = df['date'].max()
    end_date = op.augmente_heure(last_date)
    current_date = JourActuel
    
    if os.path.isfile(adrTr):
        df_tr = pd.read_csv(adrTr)
        daily_temps = {}

        for i, row in df_tr.iterrows():
            timestamp = pd.to_datetime(row["Date"])
            day = timestamp.date()
            temp_val = row["TemperatureRessentie"]
            if day not in daily_temps:
                daily_temps[day] = []
            daily_temps[day].append(temp_val)

    else:
        while current_date <= end_date:
            index = get_index(df, current_date)
            if index is not None and index < df.shape[0]:
                temp_val = reading(df, current_date, index)
                if temp_val is not None:
                    day = pd.to_datetime(current_date).date()
                    if day not in daily_temps:
                        daily_temps[day] = []
                    daily_temps[day].append(float(f'{temp_val:.2f}'))
            current_date = op.augmente_heure(current_date)
        
        ligne_dates = []
        ligne_temps = []

        for day, temps in daily_temps.items():
            ligne_temps.extend(temps)
            ligne_dates.extend([pd.to_datetime(day)] * len(temps))

        df_tr = pd.DataFrame({
            "Date": ligne_dates,
            "TemperatureRessentie": ligne_temps
        })
        df_tr.to_csv(adrTr, index=False)
        print("📦 Nouveau CSV contenant les températures ressenties créé !")

    ligne_temps = np.array([])
    for day, temps in daily_temps.items():
        dates.append(day)
        means.append(np.mean(temps))
        ligne_temps = np.append(ligne_temps, temps)

    y = np.array(means)
    norm_y = (y - y.min()) / (y.max() - y.min())

    if drawing == True:
        # Interpolation manuelle entre rouge et bleu (du haut vers le bas)
        # rouge = (1, 0, 0), bleu = (0, 0, 1)
        colors = [(val, 0, 1 - val) for val in norm_y]

        # Créer le graphique
        plt.figure(figsize=(12, 6))
        plt.scatter(dates, means, s=20, marker='o', color=colors)
        plt.plot(dates, means, alpha=0.3, linestyle='-', color='gray')
        
        # Lignes de référence
        plt.axhline(0, color=(0.2,0.2,0.2,0.5), linestyle='--')
        plt.axhline(30, color=(0.2,0.2,0.2,0.5), linestyle='--')

        cyclone_date = pd.to_datetime('2024-01-15').date()
        if cyclone_date in daily_temps:
            cyclone_mean = daily_temps[cyclone_date]
            plt.text(cyclone_date, np.mean(cyclone_mean)+1, 'Cyclone', 
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    fontsize=10,
                    color=(0.2,0.4,1))
        
        # Mise en forme
        plt.gcf().autofmt_xdate()
        plt.xlabel('Date')
        plt.ylabel('Température moyenne quotidienne (°C)')
        plt.title('Moyenne quotidienne des températures')
        plt.grid(True, alpha=0.3)
        
        plt.show()
    else:
        return ligne_temps
    
def read_m(day, n, df=df_base):
    for _ in range(n+1):
        index = get_index(df, day)
        reading(df, day, index, show=True)
        day = op.augmente_heure(day)


def extraire_donnees(df=df_base):
    # Sélectionner toutes les colonnes sauf 'date' et 'hour' si elle existe
    colonnes_a_exclure = ['date']
    if 'hour' in df.columns:
        colonnes_a_exclure.append('hour')
    
    # Créer une copie du DataFrame sans les colonnes exclues
    df_sans_date = df.drop(columns=colonnes_a_exclure)
    
    # Convertir en matrice numpy
    matrice_donnees = df_sans_date.to_numpy()
    
    return matrice_donnees[:-1]


def extraire_dates(df=df_base):
    # Extraire la colonne 'date' et convertir en numpy array
    vecteur_dates = df['date'].astype(str).str[5:7].to_numpy()
    
    return vecteur_dates[:-1]

def extraire_T_ressentie():
    # Extraire la colonne 'date' et convertir en numpy array
    vecteur_temp = draw(drawing=False)
    
    return vecteur_temp

if __name__ == '__main__':
    # print(np.shape(extraire_donnees(df_base)))
    # print(np.shape(extraire_dates(df_base)))
    # print(np.shape(extraire_T_ressentie()))
    draw()

    # read_m('2024-04-09-12:00', 0)
    # read_m('2024-08-12-12:00', 0)
    # read_m('2024-12-19-12:00', 0)
    # read_m('2024-06-23-12:00', 0)
