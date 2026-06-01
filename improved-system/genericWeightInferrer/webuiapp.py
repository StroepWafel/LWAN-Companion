import json
import math
import time
import threading
import datetime
import requests
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# --- Filepaths ---
WEIGHTS_FILE = "improved-system\\genericWeightInferrer\\longerweights.json"
NORM_STATS_FILE = "improved-system\\genericWeightInferrer\\norm_stats.json"

# --- Adelaide coordinates ---
LATITUDE = -34.9275
LONGITUDE = 138.6000

# --- Poll interval (seconds) ---
POLL_INTERVAL = 600  # 10 minutes

# --- Global state ---
latest_result = {
    "timestamp": None,
    "will_rain": None,
    "confidence": None,
    "raw_output": None,
    "weather": {}
}
result_lock = threading.Lock()

# -------------------------
# Load model
# -------------------------
def load_model():
    with open(WEIGHTS_FILE) as f:
        saved = json.load(f)
    weights = [np.array(layer["weights"]) for layer in saved["layers"]]
    biases  = [np.array(layer["bias"]) if isinstance(layer["bias"], list)
               else float(layer["bias"]) for layer in saved["layers"]]
    print(f"Loaded architecture: {saved['architecture']}")
    return weights, biases

def load_norm_stats():
    with open(NORM_STATS_FILE) as f:
        return json.load(f)

# -------------------------
# Normalization
# -------------------------
def normalize(value, col, norm_stats):
    mn = norm_stats[col]["min"]
    mx = norm_stats[col]["max"]
    if mx != mn:
        return (value - mn) / (mx - mn)
    return 0.0

# -------------------------
# Forward pass
# -------------------------
def infer(x, weights, biases):
    a = x
    for idx, (W, b) in enumerate(zip(weights, biases)):
        z = W @ a + b
        if idx < len(weights) - 1:
            a = np.maximum(0, z)           # ReLU hidden
        else:
            a = 1 / (1 + np.exp(-z))       # sigmoid output
    return float(np.squeeze(a))

# -------------------------
# Fetch weather from Open-Meteo
# -------------------------
def fetch_weather():
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    # Current conditions (forecast API — free, no key needed)
    forecast_url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={LATITUDE}&longitude={LONGITUDE}"
        "&current=temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m"
        "&timezone=Australia/Adelaide"
    )

    # Yesterday's precipitation (archive API)
    archive_url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&start_date={yesterday}&end_date={yesterday}"
        "&daily=precipitation_sum"
        "&timezone=Australia/Adelaide"
    )

    forecast_resp = requests.get(forecast_url, timeout=10).json()
    archive_resp  = requests.get(archive_url,  timeout=10).json()

    current = forecast_resp["current"]
    yesterday_precip = archive_resp["daily"]["precipitation_sum"][0] or 0.0

    return {
        "temperature":      current["temperature_2m"],
        "humidity":         current["relative_humidity_2m"],
        "pressure":         current["pressure_msl"],
        "wind_speed":       current["wind_speed_10m"],
        "rain_yesterday":   yesterday_precip,
        "date":             str(today)
    }

# -------------------------
# Run inference cycle
# -------------------------
def run_cycle(weights, biases, norm_stats):
    print(f"[{datetime.datetime.now():%H:%M:%S}] Polling weather...")
    try:
        weather = fetch_weather()

        today = datetime.date.today()
        day_of_year = today.timetuple().tm_yday
        season_sin = math.sin(2 * math.pi * day_of_year / 365.0)
        season_cos = math.cos(2 * math.pi * day_of_year / 365.0)

        x = np.array([
            normalize(weather["temperature"],    "temperature",    norm_stats),
            normalize(weather["humidity"],       "humidity",       norm_stats),
            normalize(weather["pressure"],       "pressure",       norm_stats),
            normalize(weather["wind_speed"],     "wind_speed",     norm_stats),
            season_sin,
            season_cos,
            normalize(weather["rain_yesterday"], "rain_yesterday", norm_stats),
        ])

        yhat = infer(x, weights, biases)
        will_rain = yhat >= 0.5

        # Confidence: distance from 0.5, scaled to 0-100%
        confidence = round(abs(yhat - 0.5) * 200, 1)

        with result_lock:
            latest_result.update({
                "timestamp":  datetime.datetime.now().isoformat(),
                "will_rain":  will_rain,
                "confidence": confidence,
                "raw_output": round(yhat, 4),
                "weather":    weather
            })

        print(f"  → {'RAIN' if will_rain else 'NO RAIN'} | raw={yhat:.4f} | confidence={confidence}%")

    except Exception as e:
        print(f"  ✗ Error during cycle: {e}")

# -------------------------
# Background polling thread
# -------------------------
def polling_thread(weights, biases, norm_stats):
    while True:
        run_cycle(weights, biases, norm_stats)
        time.sleep(POLL_INTERVAL)

# -------------------------
# HTTP server
# -------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silence default access log

    def send_json(self, data, status=200):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, html):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/latest":
            with result_lock:
                self.send_json(latest_result)

        elif path == "/api/status":
            with result_lock:
                has_data = latest_result["timestamp"] is not None
            self.send_json({"ok": True, "has_data": has_data, "poll_interval_seconds": POLL_INTERVAL})

        elif path == "/" or path == "/index.html":
            self.send_html(UI_HTML)

        else:
            self.send_json({"error": "Not found"}, 404)


# -------------------------
# Web UI
# -------------------------
UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Adelaide Rain Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #0a0c10;
    --surface:  #111318;
    --border:   #1e2128;
    --rain:     #4fc3f7;
    --dry:      #ffb347;
    --accent:   #4fc3f7;
    --muted:    #4a5060;
    --text:     #e8eaf0;
    --sub:      #7a8090;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Mono', monospace;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2rem;
    overflow-x: hidden;
  }

  /* animated rain background */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
      180deg,
      transparent 0px,
      transparent 3px,
      rgba(79,195,247,0.015) 3px,
      rgba(79,195,247,0.015) 4px
    );
    pointer-events: none;
    z-index: 0;
  }

  .container {
    position: relative;
    z-index: 1;
    width: 100%;
    max-width: 640px;
  }

  header {
    margin-bottom: 3rem;
  }

  .city {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(3rem, 10vw, 5.5rem);
    letter-spacing: 0.05em;
    line-height: 1;
    color: var(--text);
  }

  .city span {
    color: var(--accent);
  }

  .subtitle {
    font-size: 0.7rem;
    color: var(--sub);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.4rem;
  }

  /* main verdict card */
  .verdict-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 2.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.6s;
  }

  .verdict-card.rain  { border-color: var(--rain); }
  .verdict-card.dry   { border-color: var(--dry);  }

  .verdict-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent);
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.8s cubic-bezier(.16,1,.3,1), background 0.6s;
  }
  .verdict-card.rain::before  { background: var(--rain); transform: scaleX(1); }
  .verdict-card.dry::before   { background: var(--dry);  transform: scaleX(1); }

  .verdict-label {
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--sub);
    margin-bottom: 0.75rem;
  }

  .verdict-text {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(3rem, 12vw, 6rem);
    line-height: 1;
    letter-spacing: 0.04em;
    transition: color 0.6s;
  }
  .verdict-card.rain  .verdict-text { color: var(--rain); }
  .verdict-card.dry   .verdict-text { color: var(--dry);  }
  .verdict-card.loading .verdict-text { color: var(--muted); }

  .confidence-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 1.5rem;
  }

  .confidence-bar-wrap {
    flex: 1;
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
  }

  .confidence-bar {
    height: 100%;
    width: 0%;
    border-radius: 2px;
    transition: width 1s cubic-bezier(.16,1,.3,1), background 0.6s;
    background: var(--muted);
  }
  .verdict-card.rain .confidence-bar { background: var(--rain); }
  .verdict-card.dry  .confidence-bar { background: var(--dry);  }

  .confidence-pct {
    font-size: 0.75rem;
    color: var(--sub);
    min-width: 3rem;
    text-align: right;
  }

  .raw-output {
    font-size: 0.65rem;
    color: var(--muted);
    margin-top: 0.5rem;
  }

  /* stats grid */
  .stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 1rem;
  }

  .stat {
    background: var(--surface);
    padding: 1.25rem 1.5rem;
  }

  .stat-label {
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--sub);
    margin-bottom: 0.4rem;
  }

  .stat-value {
    font-size: 1.3rem;
    font-weight: 500;
    color: var(--text);
  }

  .stat-unit {
    font-size: 0.65rem;
    color: var(--muted);
    margin-left: 0.2rem;
  }

  /* footer row */
  .footer-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.6rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    padding: 0 0.25rem;
  }

  .dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--muted);
    margin-right: 0.4rem;
    vertical-align: middle;
    animation: pulse 2s infinite;
  }
  .dot.live { background: #4caf50; }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
  }

  .refresh-btn {
    background: none;
    border: 1px solid var(--border);
    color: var(--sub);
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    padding: 0.3rem 0.8rem;
    cursor: pointer;
    border-radius: 2px;
    text-transform: uppercase;
    transition: border-color 0.2s, color 0.2s;
  }
  .refresh-btn:hover { border-color: var(--accent); color: var(--accent); }
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="city">ADEL<span>AI</span>DE</div>
    <div class="subtitle">Neural Rain Predictor &mdash; MLP v2</div>
  </header>

  <div class="verdict-card loading" id="verdictCard">
    <div class="verdict-label">Today's Forecast</div>
    <div class="verdict-text" id="verdictText">LOADING</div>
    <div class="confidence-row">
      <div class="confidence-bar-wrap">
        <div class="confidence-bar" id="confBar"></div>
      </div>
      <div class="confidence-pct" id="confPct">—</div>
    </div>
    <div class="raw-output" id="rawOutput">raw model output: —</div>
  </div>

  <div class="stats-grid">
    <div class="stat">
      <div class="stat-label">Temperature</div>
      <div class="stat-value" id="temp">—<span class="stat-unit">°C</span></div>
    </div>
    <div class="stat">
      <div class="stat-label">Humidity</div>
      <div class="stat-value" id="humidity">—<span class="stat-unit">%</span></div>
    </div>
    <div class="stat">
      <div class="stat-label">Pressure</div>
      <div class="stat-value" id="pressure">—<span class="stat-unit">hPa</span></div>
    </div>
    <div class="stat">
      <div class="stat-label">Wind Speed</div>
      <div class="stat-value" id="wind">—<span class="stat-unit">km/h</span></div>
    </div>
    <div class="stat">
      <div class="stat-label">Yesterday's Rain</div>
      <div class="stat-value" id="rainYesterday">—<span class="stat-unit">mm</span></div>
    </div>
    <div class="stat">
      <div class="stat-label">Last Updated</div>
      <div class="stat-value" style="font-size:0.9rem" id="updated">—</div>
    </div>
  </div>

  <div class="footer-row">
    <div><span class="dot live" id="liveDot"></span>polls every 10 min &mdash; GET /api/latest for JSON</div>
    <button class="refresh-btn" onclick="fetchData()">↻ refresh</button>
  </div>
</div>

<script>
async function fetchData() {
  try {
    const r = await fetch('/api/latest');
    const d = await r.json();

    const card = document.getElementById('verdictCard');
    const vtext = document.getElementById('verdictText');
    const confBar = document.getElementById('confBar');
    const confPct = document.getElementById('confPct');
    const rawOut = document.getElementById('rawOutput');

    if (!d.timestamp) {
      vtext.textContent = 'WAITING';
      return;
    }

    const rain = d.will_rain;
    card.className = 'verdict-card ' + (rain ? 'rain' : 'dry');
    vtext.textContent = rain ? 'RAIN' : 'NO RAIN';
    confBar.style.width = d.confidence + '%';
    confPct.textContent = d.confidence + '%';
    rawOut.textContent = 'raw model output: ' + d.raw_output;

    const w = d.weather;
    document.getElementById('temp').innerHTML         = (w.temperature ?? '—') + '<span class="stat-unit">°C</span>';
    document.getElementById('humidity').innerHTML     = (w.humidity    ?? '—') + '<span class="stat-unit">%</span>';
    document.getElementById('pressure').innerHTML     = (w.pressure    ?? '—') + '<span class="stat-unit">hPa</span>';
    document.getElementById('wind').innerHTML         = (w.wind_speed  ?? '—') + '<span class="stat-unit">km/h</span>';
    document.getElementById('rainYesterday').innerHTML= (w.rain_yesterday ?? '—') + '<span class="stat-unit">mm</span>';

    const ts = new Date(d.timestamp);
    document.getElementById('updated').textContent = ts.toLocaleTimeString('en-AU', {hour:'2-digit', minute:'2-digit'});
    document.getElementById('liveDot').className = 'dot live';

  } catch(e) {
    console.error(e);
    document.getElementById('liveDot').className = 'dot';
  }
}

fetchData();
setInterval(fetchData, 30000); // refresh UI every 30s (server polls every 10m)
</script>
</body>
</html>
"""

# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    print("Loading model and norm stats...")
    weights, biases = load_model()
    norm_stats = load_norm_stats()

    print("Running initial inference cycle...")
    run_cycle(weights, biases, norm_stats)

    t = threading.Thread(target=polling_thread, args=(weights, biases, norm_stats), daemon=True)
    t.start()

    PORT = 8765
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"\nServer running at http://localhost:{PORT}")
    print(f"API endpoint:  http://localhost:{PORT}/api/latest")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")