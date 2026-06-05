import json
import numpy as np
import requests
import pandas as pd

# Adelaide coordinates
LATITUDE  = -34.92750
LONGITUDE = 138.60000

# Date range
START_DATE = "2026-01-01"
END_DATE   = "2026-05-31"

# Output
datafile      = "improved-system\\genericWeightInferrer\\hourly\\adelaide_weather_testset_hourly.csv"

url = (
    "https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={LATITUDE}"
    f"&longitude={LONGITUDE}"
    f"&start_date={START_DATE}"
    f"&end_date={END_DATE}"
    "&hourly="
    "temperature_2m,"
    "relative_humidity_2m,"
    "pressure_msl,"
    "wind_speed_10m,"
    "precipitation"
    "&timezone=Australia/Adelaide"
)

response = requests.get(url)
data     = response.json()
hourly   = data["hourly"]

df = pd.DataFrame({
    "datetime":     hourly["time"],
    "temperature":  hourly["temperature_2m"],
    "humidity":     hourly["relative_humidity_2m"],
    "pressure":     hourly["pressure_msl"],
    "wind_speed":   hourly["wind_speed_10m"],
    "precipitation":hourly["precipitation"]
})

df["datetime"] = pd.to_datetime(df["datetime"])

# Season encoding (day of year)
day_of_year = df["datetime"].dt.dayofyear
df["season_sin"] = np.sin(2 * np.pi * day_of_year / 365.0)
df["season_cos"] = np.cos(2 * np.pi * day_of_year / 365.0)

# Time of day encoding
hour = df["datetime"].dt.hour
df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

# Binary classification: did it rain this hour?
df["rain"] = (df["precipitation"] > 0.1).astype(int)

# Remove missing values
df = df.dropna()

# Last hour's precipitation as a feature
df["rain_lasthour"] = df["precipitation"].shift(1).fillna(0)

# Round trig columns
for col in ["season_sin", "season_cos", "hour_sin", "hour_cos"]:
    df[col] = df[col].round(10)

# Save dataset
df.to_csv(datafile, index=False)
print(f"Dataset saved as {datafile}")
print(f"Rows: {len(df)}")
print(df.head())