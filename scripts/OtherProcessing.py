import pandas as pd

def augmente_heure(time_str, hours=1):
    # Transforme la date en string en NumPy datetime64
    time_obj = pd.Timestamp(time_str)
    
    # Ajoute une ou plusieurs heures
    new_time_obj = time_obj + pd.Timedelta(hours=hours)
    
    # Retransforme en string
    return new_time_obj.strftime("%Y-%m-%d-%H:%M")
    


if __name__ == "__main__":
    current_time = "2024-01-01-12:00"
    next_time = augmente_heure(current_time)
    print(next_time)  # Output: "2024-01-01T13:00"

    next_time = augmente_heure(next_time, hours=11)
    print(next_time)  # Output: "2024-01-02T00:00"