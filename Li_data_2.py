import pandas as pd
from mp_api.client import MPRester

# Replace with your actual Materials Project API Key
API_KEY = "AJXBatqfDbS8CD0aL1Py9dRftUNqbifX"

def fetch_li_dataset():
    print("Connecting to Materials Project API...")
    
    with MPRester(API_KEY) as mpr:
        # Fetch intercalation electrodes specifically for Lithium (Li)
        electrodes = mpr.insertion_electrodes.search(working_ion=["Li"])
        
        data = []
        for entry in electrodes:
            data.append({
                'battery_id': entry.battery_id,
                'formula_charge': entry.formula_charge,
                'formula_discharge': entry.formula_discharge,
                'average_voltage': entry.average_voltage,
                'capacity_grav': entry.capacity_grav,
                'capacity_vol': entry.capacity_vol,
                'energy_grav': entry.energy_grav,
                'max_delta_volume': entry.max_delta_volume,  # Critical 3D structural feature
                'stability_charge': entry.stability_charge,
                'num_steps': entry.num_steps
            })

    df = pd.DataFrame(data)
    print(f"Successfully retrieved {len(df)} Li electrode entries.")

    # Save directly into the preprocessing/ folder
    output_path = 'preprocessing/raw_mp_li_v2.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved raw enriched dataset to '{output_path}'!")

if __name__ == "__main__":
    fetch_li_dataset()