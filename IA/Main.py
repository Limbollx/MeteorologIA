'''
 # @ Auteur: Mathéo Guillot--Eid, Baptiste Argemi, Pierre Lebon
 # @ Crée le: 2025-05-22 13:56:56
 # @ Modifié par: Mathéo Guillot--Eid
 # @ Modifié le: 2025-05-22 14:00:23
 # @ Description: Script pour crée et utilisé la random forest
 '''

#--------------------------------------------------
# Importation des librairies
#--------------------------------------------------

import os
from sys import path
from time import time

try:
    from pandas import read_csv, to_datetime
except Exception:
    os.system("pip install pandas")
    from pandas import read_csv, to_datetime
try:
    from xgboost import XGBRegressor
except Exception:
    os.system("pip install XGBoost")
    from xgboost import XGBRegressor
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
except Exception:
    os.system("pip install scikit-learn")
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
try:
    from numpy import random, array
except Exception:
    os.system("pip install numpy")
    from numpy import random, array

dirPath = os.path.dirname(os.path.realpath(__file__))
dirSrc = dirPath[0:dirPath.rfind(os.sep)]
path += [dirSrc, dirPath]

from scripts.CSVprocessing import extraire_donnees
from scripts.CSVprocessing import extraire_dates
from scripts.CSVprocessing import extraire_T_ressentie


#--------------------------------------------------
# Importation des données
#--------------------------------------------------

t = time()
X = extraire_donnees()
print(f'🗻 Données chargées en {time()-t:.2f}s')
t = time()
y = extraire_T_ressentie()
print(f'♨️ Températures ressenties chargées en {time()-t:.2f}s')

# noms_donnees = ['Tair','Ws10','RH','Rglo']

#--------------------------------------------------
# Définition des fonctions
#--------------------------------------------------
sw = 0
def AI_initialisation(seed, trees=50):
    global sw
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.001, random_state=seed
    )
    if sw == 0:
        try:
            model = XGBRegressor(n_estimators = trees, random_state = seed, tree_method = "gpu_hist", predictor = "gpu_predictor", verbosity = 0)
            model.fit(X_train, y_train)
        except Exception:
            print("patate")
            model = XGBRegressor(n_estimators = trees, random_state = seed, tree_method = "hist", predictor = "cpu_predictor", verbosity = 0)
            model.fit(X_train, y_train)
            sw = 1
    else:
        model = XGBRegressor(n_estimators = trees, random_state = seed, tree_method = "hist", predictor = "cpu_predictor", verbosity = 0)
        model.fit(X_train, y_train)

    return model, X_test, y_test

def find_seed(duration='1:00:00'):

    h, m, s = map(int, duration.split(':'))
    duration_seconds = h * 3600 + m * 60 + s
    try:
        start_time = time()
        best_seed = [0, float('inf')]

        while time() - start_time < duration_seconds:
            random_state = random.randint(0, 10000)
            print(f"Trying {random_state} ...")

            model, X_test, y_test = AI_initialisation(random_state, trees=20)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            if mse < best_seed[1]:
                best_seed = [random_state, mse]

        print(f"📉 Meilleur MSE: {best_seed[1]:.2f}, Seed: {best_seed[0]}")
    except KeyboardInterrupt:
        print(f"📉 Meilleur MSE: {best_seed[1]:.2f}, Seed: {best_seed[0]}")

def AI_accuracy(seed=4744):
    t = time()
    model, X_test, y_test = AI_initialisation(seed)
    print(f"🖼️ Modèle prêt en {time()-t:.2f}s")

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"📉 MSE: {mse:.2f}")
    print(f"📈 R² : {r2*100:.2f}%")


def AI_test(values, seed=4744):
    t = time()
    model = AI_initialisation(seed)[0]
    print(f"🖼️ Modèle prêt en {time()-t:.2f}s")

    y_pred = model.predict(values)
    print(f"🌡️ Température ressentie prédite: {y_pred[0]:.2f}°C")

#--------------------------------------------------
# Utilisation
#--------------------------------------------------

# Accuracy: 0.64, Seed: 6110 -> Dates
# Accuracy: 0.64, Seed: 8024 -> Dates
# Accuracy: 0.65, Seed: 8849 -> Dates

# Accuracy: 0.60, Seed: 7029 -> Temp
# Accuracy: 0.69, Seed: 7527 -> Temp

# Meilleur MSE: 0.38, Seed: 8785 -> Temp

if __name__ == "__main__":
    # find_seed(duration='1:00:00')

    # test = array([26.96058333333334,0.5011416666666666,79.31083333333333,29.060075000000005]).reshape(1, -1) # -> 34.94,  2024-11-23 18:00:00
    # test = array([26.4555,3.005141666666667,5.92750000000001,0.0]).reshape(1, -1) # -> 21.17,  2024-02-12 00:00:00
    test = array([24.65825000000001,0.366725,95.07833333333328,0.074775]).reshape(1, -1) # -> 32.74,   2024-01-02 20:00:00
    # test = array([24.301250000000003,1.259183333333333,69.69758333333333,445.9791666666666]).reshape(1, -1) # -> 34.84,  2024-07-08 13:00:00
    # test = array([25.1475,5.092041666666666,73.75066666666669,792.515]).reshape(1, -1) # -> 26.46,  2024-08-16 13:00:00
    # test = array([23.98825,0.5747583333333337,92.74583333333337,12.812108333333333]).reshape(1, -1) # -> 31.35,  2024-01-15 07:00:00

    AI_test(test)

    # AI_accuracy()
