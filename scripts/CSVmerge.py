import pandas as pd
import os

# Définition des chemins et du fichier de données
dirPath = os.path.dirname(os.path.realpath(__file__))  # Chemin du script

dirSrc = dirPath[0:dirPath.rfind(os.sep)]  # Répertoire parent

adr1 = dirSrc + os.sep + 'data' + os.sep + 'meteo_st-pierre_2024.csv'
adr2 = dirSrc + os.sep + 'data' + os.sep + 'rayonnement_st-pierre_2024_horaires.csv'

def clean_csv(file_path):
    with open(file_path, 'r') as f:
        lines = [line.rstrip(';\n') + '\n' for line in f]
    with open(file_path, 'w') as f:
        f.writelines(lines)

clean_csv(adr1)

# Load the CSV files
df1 = pd.read_csv(adr1, sep=";", parse_dates=["date"], skipinitialspace=True)
df2 = pd.read_csv(adr2, sep=";", parse_dates=["date"], skipinitialspace=True)

df2 = df2.loc[:, df2.columns.notna()]
df2 = df2.dropna(axis=1, how='all')

# Merge the dataframes on the 'date' column
merged_df = pd.merge(df1, df2, on="date", how="inner")
merged_df = merged_df.drop('Wd10', axis='columns')

# Save the merged dataframe
merged_df.to_csv(dirSrc + os.sep + "data" + os.sep + "full-data-st_pierre2-2024.csv", sep=";", index=False)

print(merged_df)
