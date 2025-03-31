#pi
import numpy as np
import matplotlib.pyplot as plt

# Paramètres du modèle
sigma = 3  # Largeur de la courbe (plus sigma est grand, plus la courbe est étalée)
A = 2  # Amplitude
t0 = 269  # Décalage de phase pour que le pic soit en juin
B = 6  # Moyenne du rayonnement solaire
t_s = 6.5       # Heure du lever du soleil
t_c = 18.5      # Heure du coucher du soleil
j = 310

# Créer un tableau d'heures de la journée (de 0 à 24 heures)
heures = np.linspace(0, 24, 1000)
jours = np.arange(1, 365)

# Calculer le rayonnement solaire pour chaque jour
max_rayonnement = (A * np.sin(2*np.pi/365 * (jours - t0)) + B)/8


# Fonction de rayonnement solaire
def solar_irradiance(t, t_s, t_c):
    if t_s <= t <= t_c:
        return np.sin(np.pi * (t - t_s) / (t_c - t_s))
    else:
        return 0

# Génération des données
I_values = [solar_irradiance(t, t_s, t_c) for t in heures]

if __name__ == "__main__":
    print(max(I_values))

    # Tracé du modèle
    plt.figure(figsize=(10, 5))
    plt.scatter(12.5, 0.14, marker='o', s=200, c='r')
    plt.scatter(12.5, 0.03, marker='2', s=1000, c='r')
    plt.scatter(12.5, 0.09, marker='1', s=1000, c='r')
    plt.plot(heures, I_values, label="Rayonnement solaire (modèle)", color="orange")
    plt.xlabel("Heure de la journée")
    plt.ylabel("Rayonnement solaire (kW)")
    plt.title("Modélisation du rayonnement solaire en fonction du temps")
    plt.legend()
    plt.grid()
    plt.show()