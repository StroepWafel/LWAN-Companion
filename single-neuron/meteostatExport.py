import requests
import pandas as pd

# Adelaide coordinates
LATITUDE = -34.92750
LONGITUDE = 138.60000

# Date range
START_DATE = "2023-12-31"
END_DATE = "2025-12-31"

# Output
datafile = "adelaide_weather_dataset.csv"

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

# Convert to Binary classification 
# 1 = rain, 0 = no rain
df["rain"] = (df["precipitation"] > 0.2).astype(int)

# Remove missing values
df = df.dropna()


# Normalize numerical columns
feature_columns = [
    "temperature",
    "humidity",
    "pressure",
    "wind_speed"
]

for col in feature_columns:
    min_val = df[col].min()
    max_val = df[col].max()

    df[col] = (df[col] - min_val) / (max_val - min_val)


# Print small sample
print(df.head())

# Save dataset
df.to_csv(datafile, index=False)

print(f"\nDataset saved as {datafile}")
print(f"Rows: {len(df)}")