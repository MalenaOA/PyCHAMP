import pandas as pd
import os

def get_wd():
    return os.path.dirname(os.path.abspath(__file__))
wd = get_wd()

# Load the CSV files
pyaqua_df_full = pd.read_csv('/Users/michellenguyen/Downloads/pyaqua_df_full.csv')
df_aqua_units = pd.read_csv(wd + '/Outputs/df_aqua_units_20240818_222323.csv')

# Convert bias corrected yield from t/ha to bu/ha
pyaqua_df_full['Bias Corrected Yield (bu/ha)'] = pyaqua_df_full['Bias Corrected Yield (t/ha)'] / 0.0254

# Initialize an empty list to store rows before converting to DataFrame
rows = []

# Loop through df_aqua_units and fill in the new DataFrame
for index, row in df_aqua_units.iterrows():
    if row['crop'] == 'others' or row['irrig_method'] == 0:
        yield_aquacrop_bu = 'N/A'
    else:
        # Find the corresponding bias corrected yield
        matching_row = pyaqua_df_full[(pyaqua_df_full['bid'] == row['bid']) & (pyaqua_df_full['year'] == row['year'])]
        if not matching_row.empty:
            yield_aquacrop_bu = matching_row.iloc[0]['Bias Corrected Yield (bu/ha)']
        else:
            yield_aquacrop_bu = 'N/A'
    
    # Append the information to the list
    rows.append({
        'year': row['year'],
        'bid': row['bid'],
        'maxirr_season': row['maxirr_season'],
        'crop': row['crop'],
        'irrig_method': row['irrig_method'],
        'yield_pychamp_bu': row['yield_pychamp_bu'],
        'yield_aquacrop_bu': yield_aquacrop_bu
    })

# Convert the list of rows to a DataFrame
new_df = pd.DataFrame(rows)

output_dir = os.path.join(wd, f"Outputs")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the new DataFrame to a CSV file
new_df.to_csv(os.path.join(output_dir, f'df_aqua_units_yields_combined.csv'), index=True)

print("New CSV file created with bias corrected yield information!")
