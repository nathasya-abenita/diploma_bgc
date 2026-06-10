import xarray as xr
import pandas as pd
from scipy.signal import detrend
from dateutil.relativedelta import relativedelta

# Open data
ds = xr.open_dataset('./data/iersst_nino3.4a_rel.nc', decode_times=False)

# Extract base date from units
units = ds.time.attrs["units"]  # "months since 1854-01-15"
_, _, base_str = units.partition("since ")
base = pd.Timestamp(base_str.strip())

# Convert numeric months to real timestamps
months = ds.time.values.astype(int)
decoded = [base + relativedelta(months=m) for m in months]

# Assign back
ds = ds.assign_coords(time=("time", decoded))

# Convert to DataFrame
df = ds.to_dataframe()
df = df.rename(columns={'Nino3.4r': 'enso'})
df = df.dropna()['1950': '2025']
df['enso'] = detrend(df['enso'], type='linear')

# Save
df.to_csv('./data/nino34r_det.csv')