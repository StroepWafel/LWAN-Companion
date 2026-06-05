import numpy as np
import requests
import pandas as pd

# Adelaide coordinates
LATITUDE = -34.92750
LONGITUDE = 138.60000

# Date range
START_DATE = "2026-01-01"
END_DATE = "2026-05-31"

# Output
datafile = "MLP\\adelaide_weather_testset.csv"

url = (
    "https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={LATITUDE}"
    f"&longitude={LONGITUDE}"
    f"&start_date={START_DATE}"
    f"&end_date={END_DATE}"
    "&daily="
    "temperature_2m_mean,"
    "relative_humidity_2m_mean,"
    "pressure_msl_mean,"
    "wind_speed_10m_mean,"
    "precipitation_sum"
    "&timezone=Australia/Adelaide"
)

response = requests.get(url)
data = response.json()

daily = data["daily"]

df = pd.DataFrame({
    "date": daily["time"],
    "temperature": daily["temperature_2m_mean"],
    "humidity": daily["relative_humidity_2m_mean"],
    "pressure": daily["pressure_msl_mean"],
    "wind_speed": daily["wind_speed_10m_mean"],
    "precipitation": daily["precipitation_sum"]
})

df["date"] = pd.to_datetime(df["date"])

day_of_year = df["date"].dt.dayofyear

df["season_sin"] = np.sin(2 * np.pi * day_of_year / 365.0)
df["season_cos"] = np.cos(2 * np.pi * day_of_year / 365.0)

# Binary classification
df["rain"] = (df["precipitation"] > 0.2).astype(int)

# Remove missing values
df = df.dropna()


# Print small sample
print(df.head())

df["season_sin"] = df["season_sin"].round(10)
df["season_cos"] = df["season_cos"].round(10)

# Save dataset
df.to_csv(datafile, index=False)

print(f"\nDataset saved as {datafile}")
print(f"Rows: {len(df)}")