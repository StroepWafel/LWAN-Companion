import json
import math
import time
import threading
import datetime
import requests
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available, hourly model disabled")

# --- Filepaths ---
DAILY_WEIGHTS_FILE   = "improved-system\\genericWeightInferrer\\longerweights.json"
DAILY_NORM_FILE      = "improved-system\\genericWeightInferrer\\norm_stats.json"
HOURLY_WEIGHTS_FILE  = "improved-system\\genericWeightInferrer\\longerweights_hourly.pt"
HOURLY_NORM_FILE     = "improved-system\\genericWeightInferrer\\norm_stats_hourly.json"

# --- Adelaide coordinates ---
LATITUDE  = -34.9275
LONGITUDE = 138.6000

# --- Feature flags ---
SHOW_MEASURED_RAINFALL = False

# --- Global state ---
latest_daily = {
    "timestamp": None, "will_rain": None,
    "confidence": None, "raw_output": None, "weather": {}
}
latest_hourly = {"timestamp": None, "hourly": []}
state_lock = threading.Lock()

# -------------------------
# Daily model (numpy)
# -------------------------
def load_daily_model():
    with open(DAILY_WEIGHTS_FILE) as f:
        saved = json.load(f)
    weights = [np.array(l["weights"]) for l in saved["layers"]]
    biases  = [np.array(l["bias"]) if isinstance(l["bias"], list)
               else float(l["bias"]) for l in saved["layers"]]
    print(f"Daily model loaded: {saved['architecture']}")
    return weights, biases

def load_daily_norm():
    with open(DAILY_NORM_FILE) as f:
        return json.load(f)

def norm_daily(v, col, ns):
    mn, mx = ns[col]["min"], ns[col]["max"]
    return (v - mn) / (mx - mn) if mx != mn else 0.0

def infer_daily(x, weights, biases):
    a = x
    for idx, (W, b) in enumerate(zip(weights, biases)):
        z = W @ a + b
        a = np.maximum(0, z) if idx < len(weights)-1 else 1/(1+np.exp(-z))
    return float(np.squeeze(a))

# -------------------------
# Hourly model (PyTorch)
# -------------------------
def build_hourly_model(device):
    return nn.Sequential(
        nn.Linear(9, 16), nn.ReLU(),
        nn.Linear(16,12), nn.ReLU(),
        nn.Linear(12, 8), nn.ReLU(),
        nn.Linear(8,  4), nn.ReLU(),
        nn.Linear(4,  1), nn.Sigmoid()
    ).to(device)

def load_hourly_model():
    if not TORCH_AVAILABLE:
        return None, None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_hourly_model(device)
    model.load_state_dict(torch.load(HOURLY_WEIGHTS_FILE, map_location=device))
    model.eval()
    with open(HOURLY_NORM_FILE) as f:
        ns = json.load(f)
    print(f"Hourly model loaded on {device}")
    return model, ns, device

def norm_hourly(v, col, ns):
    mn, mx = ns[col]["min"], ns[col]["max"]
    return (v - mn) / (mx - mn) if mx != mn else 0.0

def infer_hourly_all(model, ns, device, base_weather):
    today = datetime.date.today()
    doy   = today.timetuple().tm_yday
    results = []
    prev_precip = base_weather["rain_yesterday"]

    with torch.no_grad():
        for hour in range(24):
            hour_sin = math.sin(2 * math.pi * hour / 24)
            hour_cos = math.cos(2 * math.pi * hour / 24)
            # simple diurnal temperature drift
            temp_offset = math.sin((hour - 14) / 24 * 2 * math.pi) * 2.5

            x = torch.tensor([[
                norm_hourly(base_weather["temperature"] + temp_offset, "temperature", ns),
                norm_hourly(base_weather["humidity"],    "humidity",    ns),
                norm_hourly(base_weather["pressure"],    "pressure",    ns),
                norm_hourly(base_weather["wind_speed"],  "wind_speed",  ns),
                math.sin(2 * math.pi * doy / 365.0),
                math.cos(2 * math.pi * doy / 365.0),
                hour_sin,
                hour_cos,
                norm_hourly(prev_precip, "rain_lasthour", ns),
            ]], dtype=torch.float32).to(device)

            yhat = model(x).item()
            results.append({
                "hour":       hour,
                "label":      f"{hour:02d}:00",
                "prob":       round(yhat, 4),
                "rain":       yhat >= 0.5,
                "confidence": round(abs(yhat - 0.5) * 200, 1)
            })
            # chain: if model predicts rain this hour, next hour sees some precip
            prev_precip = yhat * 2.0 if yhat >= 0.5 else 0.0

    return results

# -------------------------
# Weather fetch
# -------------------------
def fetch_weather():
    today     = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    daily_param = "&daily=precipitation_sum" if SHOW_MEASURED_RAINFALL else ""

    forecast_url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={LATITUDE}&longitude={LONGITUDE}"
        "&current=temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m"
        + daily_param +
        "&timezone=Australia/Adelaide"
    )
    archive_url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&start_date={yesterday}&end_date={yesterday}"
        "&daily=precipitation_sum"
        "&timezone=Australia/Adelaide"
    )

    fr = requests.get(forecast_url, timeout=10).json()
    ar = requests.get(archive_url,  timeout=10).json()

    cur   = fr["current"]
    ydpre = ar["daily"]["precipitation_sum"][0] or 0.0
    return {
        "temperature":    cur["temperature_2m"],
        "humidity":       cur["relative_humidity_2m"],
        "pressure":       cur["pressure_msl"],
        "wind_speed":     cur["wind_speed_10m"],
        "rain_yesterday": ydpre,
        "rain_today":     (fr["daily"]["precipitation_sum"][0] or 0.0) if SHOW_MEASURED_RAINFALL else None,
        "date":           str(today)
    }

# -------------------------
# Inference cycles
# -------------------------
def run_daily_cycle(d_weights, d_biases, d_ns):
    print(f"[{datetime.datetime.now():%H:%M:%S}] Daily poll...")
    try:
        weather    = fetch_weather()
        doy        = datetime.date.today().timetuple().tm_yday
        season_sin = math.sin(2 * math.pi * doy / 365.0)
        season_cos = math.cos(2 * math.pi * doy / 365.0)

        x = np.array([
            norm_daily(weather["temperature"],    "temperature",    d_ns),
            norm_daily(weather["humidity"],       "humidity",       d_ns),
            norm_daily(weather["pressure"],       "pressure",       d_ns),
            norm_daily(weather["wind_speed"],     "wind_speed",     d_ns),
            season_sin, season_cos,
            norm_daily(weather["rain_yesterday"], "rain_yesterday", d_ns),
        ])

        yhat       = infer_daily(x, d_weights, d_biases)
        will_rain  = bool(yhat >= 0.5)
        confidence = round(abs(yhat - 0.5) * 200, 1)

        with state_lock:
            latest_daily.update({
                "timestamp": datetime.datetime.now().isoformat(),
                "will_rain": will_rain, "confidence": confidence,
                "raw_output": round(yhat, 4), "weather": weather
            })
        print(f"  Daily → {'RAIN' if will_rain else 'NO RAIN'} | raw={yhat:.4f} | conf={confidence}%")
    except Exception as e:
        print(f"  Daily error: {e}")

def run_current_hour(h_model, h_ns, h_device, weather):
    """Run inference for the current hour only — called every 10 minutes."""
    if h_model is None:
        return
    try:
        now  = datetime.datetime.now()
        doy  = now.timetuple().tm_yday
        h    = now.hour
        prev_precip = weather["rain_yesterday"]

        x = torch.tensor([[
            norm_hourly(weather["temperature"], "temperature", h_ns),
            norm_hourly(weather["humidity"],    "humidity",    h_ns),
            norm_hourly(weather["pressure"],    "pressure",    h_ns),
            norm_hourly(weather["wind_speed"],  "wind_speed",  h_ns),
            math.sin(2 * math.pi * doy / 365.0),
            math.cos(2 * math.pi * doy / 365.0),
            math.sin(2 * math.pi * h   / 24.0),
            math.cos(2 * math.pi * h   / 24.0),
            norm_hourly(prev_precip, "rain_lasthour", h_ns),
        ]], dtype=torch.float32).to(h_device)

        with torch.no_grad():
            yhat = h_model(x).item()

        with state_lock:
            latest_hourly["timestamp"]  = now.isoformat()
            latest_hourly["hour"]       = h
            latest_hourly["label"]      = f"{h:02d}:00"
            latest_hourly["will_rain"]  = bool(yhat >= 0.5)
            latest_hourly["confidence"] = round(abs(yhat - 0.5) * 200, 1)
            latest_hourly["raw_output"] = round(yhat, 4)
            latest_hourly["weather"]    = weather
            # Update just offset=0 in forecast if it exists
            if latest_hourly.get("forecast"):
                latest_hourly["forecast"][0] = {
                    "offset": 0, "hour": h, "label": f"{h:02d}:00",
                    "will_rain": bool(yhat >= 0.5),
                    "confidence": round(abs(yhat - 0.5) * 200, 1),
                    "raw_output": round(yhat, 4)
                }
        print(f"  Current hour {h:02d}:00 | {'RAIN' if yhat>=0.5 else 'NO RAIN'} | raw={yhat:.4f}")
    except Exception as e:
        print(f"  Current hour error: {e}")


def run_forecast(h_model, h_ns, h_device, weather):
    """Run 5-hour lookahead — called once per hour."""
    if h_model is None:
        return
    try:
        now         = datetime.datetime.now()
        forecast    = []
        prev_precip = weather["rain_yesterday"]

        with torch.no_grad():
            for offset in range(6):  # current + 5 ahead
                h          = (now.hour + offset) % 24
                future_day = now + datetime.timedelta(hours=offset)
                fdoy       = future_day.timetuple().tm_yday

                x = torch.tensor([[
                    norm_hourly(weather["temperature"], "temperature", h_ns),
                    norm_hourly(weather["humidity"],    "humidity",    h_ns),
                    norm_hourly(weather["pressure"],    "pressure",    h_ns),
                    norm_hourly(weather["wind_speed"],  "wind_speed",  h_ns),
                    math.sin(2 * math.pi * fdoy / 365.0),
                    math.cos(2 * math.pi * fdoy / 365.0),
                    math.sin(2 * math.pi * h    / 24.0),
                    math.cos(2 * math.pi * h    / 24.0),
                    norm_hourly(prev_precip, "rain_lasthour", h_ns),
                ]], dtype=torch.float32).to(h_device)

                yhat = h_model(x).item()
                forecast.append({
                    "offset":     offset,
                    "hour":       h,
                    "label":      f"{h:02d}:00",
                    "will_rain":  bool(yhat >= 0.5),
                    "confidence": round(abs(yhat - 0.5) * 200, 1),
                    "raw_output": round(yhat, 4)
                })
                prev_precip = yhat * 2.0 if yhat >= 0.5 else 0.0

        with state_lock:
            latest_hourly["forecast"] = forecast
        print(f"  Forecast updated: {[f['label'] for f in forecast]}")
    except Exception as e:
        print(f"  Forecast error: {e}")


def run_hourly_cycle(h_model, h_ns, h_device, weather):
    """Run both current hour and forecast — used on startup."""
    run_current_hour(h_model, h_ns, h_device, weather)
    run_forecast(h_model, h_ns, h_device, weather)


def run_all_cycles(d_weights, d_biases, d_ns, h_model, h_ns, h_device):
    run_daily_cycle(d_weights, d_biases, d_ns)
    with state_lock:
        weather = latest_daily.get("weather") or {}
    if not weather:
        weather = fetch_weather()
    run_hourly_cycle(h_model, h_ns, h_device, weather)

def polling_thread(d_weights, d_biases, d_ns, h_model, h_ns, h_device):
    last_forecast_hour = -1
    while True:
        time.sleep(600)  # wake every 10 minutes
        now = datetime.datetime.now()

        # Always refresh current conditions + current-hour prediction
        run_daily_cycle(d_weights, d_biases, d_ns)
        with state_lock:
            weather = latest_daily.get("weather") or {}
        if not weather:
            weather = fetch_weather()
        run_current_hour(h_model, h_ns, h_device, weather)

        # Only re-run full 5-hour forecast when the hour changes
        if now.hour != last_forecast_hour:
            run_forecast(h_model, h_ns, h_device, weather)
            last_forecast_hour = now.hour

# -------------------------
# HTTP handler
# -------------------------
class Handler(BaseHTTPRequestHandler):
    daily_model_args  = None
    hourly_model_args = None

    def log_message(self, fmt, *args): pass

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
        path = urlparse(self.path).path

        if path == "/api/latest":
            with state_lock:
                self.send_json(latest_daily)
        elif path == "/api/hourly":
            with state_lock:
                self.send_json(latest_hourly)
        elif path == "/api/status":
            with state_lock:
                self.send_json({
                    "ok": True,
                    "daily_ready":  latest_daily["timestamp"]  is not None,
                    "hourly_ready": latest_hourly["timestamp"] is not None,
                    "hourly_model": TORCH_AVAILABLE
                })
        elif path == "/hourly":
            self.send_html(HOURLY_HTML)
        elif path in ("/", "/index.html"):
            self.send_html(DAILY_HTML)
        else:
            self.send_json({"error": "Not found"}, 404)


# -------------------------
# Daily page HTML
# -------------------------
DAILY_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Adelaide Rain Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  :root{--bg:#0a0c10;--surface:#111318;--border:#1e2128;--rain:#4fc3f7;--dry:#ffb347;--accent:#4fc3f7;--muted:#4a5060;--text:#e8eaf0;--sub:#7a8090}
  body{background:var(--bg);color:var(--text);font-family:'DM Mono',monospace;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:2rem;overflow-x:hidden}
  body::before{content:'';position:fixed;inset:0;background:repeating-linear-gradient(180deg,transparent 0px,transparent 3px,rgba(79,195,247,0.015) 3px,rgba(79,195,247,0.015) 4px);pointer-events:none;z-index:0}
  .container{position:relative;z-index:1;width:100%;max-width:640px}
  header{margin-bottom:3rem}
  .city{font-family:'Bebas Neue',sans-serif;font-size:clamp(3rem,10vw,5.5rem);letter-spacing:0.05em;line-height:1}
  .city span{color:var(--accent)}
  .subtitle{font-size:0.7rem;color:var(--sub);letter-spacing:0.2em;text-transform:uppercase;margin-top:0.4rem}
  .nav{display:flex;gap:0.5rem;margin-bottom:1.5rem}
  .nav a{font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;color:var(--sub);text-decoration:none;border:1px solid var(--border);padding:0.3rem 0.8rem;border-radius:2px;transition:border-color 0.2s,color 0.2s}
  .nav a:hover,.nav a.active{border-color:var(--accent);color:var(--accent)}
  .verdict-card{background:var(--surface);border:1px solid var(--border);border-radius:2px;padding:2.5rem;margin-bottom:1rem;position:relative;overflow:hidden;transition:border-color 0.6s}
  .verdict-card.rain{border-color:var(--rain)}.verdict-card.dry{border-color:var(--dry)}
  .verdict-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--accent);transform:scaleX(0);transform-origin:left;transition:transform 0.8s cubic-bezier(.16,1,.3,1),background 0.6s}
  .verdict-card.rain::before{background:var(--rain);transform:scaleX(1)}.verdict-card.dry::before{background:var(--dry);transform:scaleX(1)}
  .verdict-label{font-size:0.65rem;letter-spacing:0.25em;text-transform:uppercase;color:var(--sub);margin-bottom:0.75rem}
  .verdict-text{font-family:'Bebas Neue',sans-serif;font-size:clamp(3rem,12vw,6rem);line-height:1;letter-spacing:0.04em;transition:color 0.6s}
  .verdict-card.rain .verdict-text{color:var(--rain)}.verdict-card.dry .verdict-text{color:var(--dry)}.verdict-card.loading .verdict-text{color:var(--muted)}
  .confidence-row{display:flex;align-items:center;gap:1rem;margin-top:1.5rem}
  .confidence-bar-wrap{flex:1;height:3px;background:var(--border);border-radius:2px;overflow:hidden}
  .confidence-bar{height:100%;width:0%;border-radius:2px;transition:width 1s cubic-bezier(.16,1,.3,1),background 0.6s;background:var(--muted)}
  .verdict-card.rain .confidence-bar{background:var(--rain)}.verdict-card.dry .confidence-bar{background:var(--dry)}
  .confidence-pct{font-size:0.75rem;color:var(--sub);min-width:3rem;text-align:right}
  .raw-output{font-size:0.65rem;color:var(--muted);margin-top:0.5rem}
  .stats-grid{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--border);border:1px solid var(--border);border-radius:2px;overflow:hidden;margin-bottom:1rem}
  .stat{background:var(--surface);padding:1.25rem 1.5rem}
  .stat-label{font-size:0.6rem;letter-spacing:0.2em;text-transform:uppercase;color:var(--sub);margin-bottom:0.4rem}
  .stat-value{font-size:1.3rem;font-weight:500;color:var(--text)}
  .stat-unit{font-size:0.65rem;color:var(--muted);margin-left:0.2rem}
  .footer-row{display:flex;justify-content:space-between;align-items:center;font-size:0.6rem;color:var(--muted);letter-spacing:0.1em;padding:0 0.25rem}
  .dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--muted);margin-right:0.4rem;vertical-align:middle;animation:pulse 2s infinite}
  .dot.live{background:#4caf50}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
  .refresh-btn{background:none;border:1px solid var(--border);color:var(--sub);font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;padding:0.3rem 0.8rem;cursor:pointer;border-radius:2px;text-transform:uppercase;transition:border-color 0.2s,color 0.2s}
  .refresh-btn:hover{border-color:var(--accent);color:var(--accent)}
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="city">ADEL<span>AI</span>DE</div>
    <div class="subtitle">Neural Rain Predictor &mdash; MLP v2</div>
  </header>
  <nav class="nav">
    <a href="/" class="active">Daily</a>
    <a href="/hourly">Hourly</a>
  </nav>
  <div class="verdict-card loading" id="verdictCard">
    <div class="verdict-label">Today's Forecast</div>
    <div class="verdict-text" id="verdictText">LOADING</div>
    <div class="confidence-row">
      <div class="confidence-bar-wrap"><div class="confidence-bar" id="confBar"></div></div>
      <div class="confidence-pct" id="confPct">—</div>
    </div>
    <div class="raw-output" id="rawOutput">raw model output: —</div>
  </div>
  <div class="stats-grid">
    <div class="stat"><div class="stat-label">Temperature</div><div class="stat-value" id="temp">—<span class="stat-unit">°C</span></div></div>
    <div class="stat"><div class="stat-label">Humidity</div><div class="stat-value" id="humidity">—<span class="stat-unit">%</span></div></div>
    <div class="stat"><div class="stat-label">Pressure</div><div class="stat-value" id="pressure">—<span class="stat-unit">hPa</span></div></div>
    <div class="stat"><div class="stat-label">Wind Speed</div><div class="stat-value" id="wind">—<span class="stat-unit">km/h</span></div></div>
    <div class="stat"><div class="stat-label">Yesterday's Rain</div><div class="stat-value" id="rainYesterday">—<span class="stat-unit">mm</span></div></div>
    <div class="stat"><div class="stat-label">Last Updated</div><div class="stat-value" style="font-size:0.9rem" id="updated">—</div></div>
  </div>
  <div class="footer-row">
    <div><span class="dot live" id="liveDot"></span>updates at 07:00 daily &mdash; GET /api/latest</div>
    <button class="refresh-btn" onclick="fetchData()">&#x21BB; refresh</button>
  </div>
</div>
<script>
async function fetchData() {
  try {
    const d = await fetch('/api/latest').then(r=>r.json());
    const card=document.getElementById('verdictCard'),vtext=document.getElementById('verdictText');
    const confBar=document.getElementById('confBar'),confPct=document.getElementById('confPct');
    if (!d.timestamp){vtext.textContent='WAITING';return;}
    card.className='verdict-card '+(d.will_rain?'rain':'dry');
    vtext.textContent=d.will_rain?'RAIN':'NO RAIN';
    confBar.style.width=d.confidence+'%';confPct.textContent=d.confidence+'%';
    document.getElementById('rawOutput').textContent='raw model output: '+d.raw_output;
    const w=d.weather;
    document.getElementById('temp').innerHTML=(w.temperature??'—')+'<span class="stat-unit">°C</span>';
    document.getElementById('humidity').innerHTML=(w.humidity??'—')+'<span class="stat-unit">%</span>';
    document.getElementById('pressure').innerHTML=(w.pressure??'—')+'<span class="stat-unit">hPa</span>';
    document.getElementById('wind').innerHTML=(w.wind_speed??'—')+'<span class="stat-unit">km/h</span>';
    document.getElementById('rainYesterday').innerHTML=(w.rain_yesterday??'—')+'<span class="stat-unit">mm</span>';
    const ts=new Date(d.timestamp);
    document.getElementById('updated').textContent=ts.toLocaleTimeString('en-AU',{hour:'2-digit',minute:'2-digit'});
    document.getElementById('liveDot').className='dot live';
  } catch(e){document.getElementById('liveDot').className='dot';}
}
fetchData();setInterval(fetchData,30000);
</script>
</body></html>"""


# -------------------------
# Hourly page HTML
# -------------------------
HOURLY_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Adelaide Rain — This Hour</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  :root{--bg:#0a0c10;--surface:#111318;--border:#1e2128;--rain:#4fc3f7;--dry:#ffb347;--accent:#4fc3f7;--muted:#4a5060;--text:#e8eaf0;--sub:#7a8090}
  body{background:var(--bg);color:var(--text);font-family:'DM Mono',monospace;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:2rem;overflow-x:hidden}
  body::before{content:'';position:fixed;inset:0;background:repeating-linear-gradient(180deg,transparent 0px,transparent 3px,rgba(79,195,247,0.015) 3px,rgba(79,195,247,0.015) 4px);pointer-events:none;z-index:0}
  .container{position:relative;z-index:1;width:100%;max-width:640px}
  header{margin-bottom:2rem}
  .city{font-family:'Bebas Neue',sans-serif;font-size:clamp(3rem,10vw,5.5rem);letter-spacing:0.05em;line-height:1}
  .city span{color:var(--accent)}
  .subtitle{font-size:0.7rem;color:var(--sub);letter-spacing:0.2em;text-transform:uppercase;margin-top:0.4rem}
  .nav{display:flex;gap:0.5rem;margin-bottom:1.5rem}
  .nav a{font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;color:var(--sub);text-decoration:none;border:1px solid var(--border);padding:0.3rem 0.8rem;border-radius:2px;transition:border-color 0.2s,color 0.2s}
  .nav a:hover,.nav a.active{border-color:var(--accent);color:var(--accent)}
  /* hour badge */
  .hour-badge{display:inline-flex;align-items:center;gap:0.6rem;background:var(--surface);border:1px solid var(--border);border-radius:2px;padding:0.5rem 1rem;margin-bottom:1rem;font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;color:var(--sub)}
  .hour-badge .clock{font-family:'Bebas Neue',sans-serif;font-size:1.4rem;color:var(--text);letter-spacing:0.05em;line-height:1}
  .hour-badge .pip{width:6px;height:6px;border-radius:50%;background:#4caf50;animation:pulse 2s infinite;flex-shrink:0}
  /* verdict card */
  .verdict-card{background:var(--surface);border:1px solid var(--border);border-radius:2px;padding:2.5rem;margin-bottom:1rem;position:relative;overflow:hidden;transition:border-color 0.6s}
  .verdict-card.rain{border-color:var(--rain)}.verdict-card.dry{border-color:var(--dry)}
  .verdict-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--accent);transform:scaleX(0);transform-origin:left;transition:transform 0.8s cubic-bezier(.16,1,.3,1),background 0.6s}
  .verdict-card.rain::before{background:var(--rain);transform:scaleX(1)}.verdict-card.dry::before{background:var(--dry);transform:scaleX(1)}
  .verdict-label{font-size:0.65rem;letter-spacing:0.25em;text-transform:uppercase;color:var(--sub);margin-bottom:0.75rem}
  .verdict-text{font-family:'Bebas Neue',sans-serif;font-size:clamp(3rem,12vw,6rem);line-height:1;letter-spacing:0.04em;transition:color 0.6s}
  .verdict-card.rain .verdict-text{color:var(--rain)}.verdict-card.dry .verdict-text{color:var(--dry)}.verdict-card.loading .verdict-text{color:var(--muted)}
  .confidence-row{display:flex;align-items:center;gap:1rem;margin-top:1.5rem}
  .confidence-bar-wrap{flex:1;height:3px;background:var(--border);border-radius:2px;overflow:hidden}
  .confidence-bar{height:100%;width:0%;border-radius:2px;transition:width 1s cubic-bezier(.16,1,.3,1),background 0.6s;background:var(--muted)}
  .verdict-card.rain .confidence-bar{background:var(--rain)}.verdict-card.dry .confidence-bar{background:var(--dry)}
  .confidence-pct{font-size:0.75rem;color:var(--sub);min-width:3rem;text-align:right}
  .raw-output{font-size:0.65rem;color:var(--muted);margin-top:0.5rem}
  /* stats grid */
  .stats-grid{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--border);border:1px solid var(--border);border-radius:2px;overflow:hidden;margin-bottom:1rem}
  .stat{background:var(--surface);padding:1.25rem 1.5rem}
  .stat-label{font-size:0.6rem;letter-spacing:0.2em;text-transform:uppercase;color:var(--sub);margin-bottom:0.4rem}
  .stat-value{font-size:1.3rem;font-weight:500;color:var(--text)}
  .stat-unit{font-size:0.65rem;color:var(--muted);margin-left:0.2rem}
  /* forecast chart */
  .chart-card{background:var(--surface);border:1px solid var(--border);border-radius:2px;padding:1.5rem;margin-bottom:1rem}
  .chart-label{font-size:0.6rem;letter-spacing:0.2em;text-transform:uppercase;color:var(--sub);margin-bottom:1rem}
  .chart-wrap{position:relative;height:140px;margin-bottom:0.5rem}
  .midline{position:absolute;left:0;right:0;top:50%;height:1px;background:var(--sub);opacity:0.35;z-index:1}
  .bars{position:absolute;inset:0;display:flex;gap:6px;align-items:stretch}
  .bar-col{flex:1;position:relative}
  /* rain bar: pinned to bottom of top half, grows upward */
  .rain-bar{position:absolute;bottom:50%;left:0;right:0;border-radius:2px 2px 0 0;transition:height 0.7s cubic-bezier(.16,1,.3,1)}
  /* dry bar: pinned to top of bottom half, grows downward */
  .dry-bar{position:absolute;top:50%;left:0;right:0;border-radius:0 0 2px 2px;transition:height 0.7s cubic-bezier(.16,1,.3,1)}
  .rain-bar{background:var(--rain)}
  .dry-bar{background:var(--muted)}
  .bar-col.current .rain-bar{background:#7ee8fa}
  .bar-col.current .dry-bar{background:#d4a030}
  .bar-conf{position:absolute;left:0;right:0;font-size:0.5rem;color:rgba(255,255,255,0.85);text-align:center;pointer-events:none;z-index:2}
  .bar-labels{display:flex;gap:6px}
  .bar-time{flex:1;font-size:0.55rem;color:var(--sub);text-align:center;white-space:nowrap}
  .bar-col.current + .bar-time, .bar-time.current{color:var(--text)}
  /* footer */
  .footer-row{display:flex;justify-content:space-between;align-items:center;font-size:0.6rem;color:var(--muted);letter-spacing:0.1em;padding:0 0.25rem}
  .dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--muted);margin-right:0.4rem;vertical-align:middle;animation:pulse 2s infinite}
  .dot.live{background:#4caf50}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
  .refresh-btn{background:none;border:1px solid var(--border);color:var(--sub);font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;padding:0.3rem 0.8rem;cursor:pointer;border-radius:2px;text-transform:uppercase;transition:border-color 0.2s,color 0.2s}
  .refresh-btn:hover{border-color:var(--accent);color:var(--accent)}
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="city">ADEL<span>AI</span>DE</div>
    <div class="subtitle">Neural Rain Predictor &mdash; This Hour</div>
  </header>
  <nav class="nav">
    <a href="/">Daily</a>
    <a href="/hourly" class="active">This Hour</a>
  </nav>

  <div class="hour-badge">
    <div class="pip"></div>
    <div>Now &mdash;</div>
    <div class="clock" id="clockLabel">--:00</div>
  </div>

  <div class="verdict-card loading" id="verdictCard">
    <div class="verdict-label">Rain This Hour?</div>
    <div class="verdict-text" id="verdictText">LOADING</div>
    <div class="confidence-row">
      <div class="confidence-bar-wrap"><div class="confidence-bar" id="confBar"></div></div>
      <div class="confidence-pct" id="confPct">—</div>
    </div>
    <div class="raw-output" id="rawOutput">raw model output: —</div>
  </div>

  <div class="stats-grid">
    <div class="stat"><div class="stat-label">Temperature</div><div class="stat-value" id="temp">—<span class="stat-unit">°C</span></div></div>
    <div class="stat"><div class="stat-label">Humidity</div><div class="stat-value" id="humidity">—<span class="stat-unit">%</span></div></div>
    <div class="stat"><div class="stat-label">Pressure</div><div class="stat-value" id="pressure">—<span class="stat-unit">hPa</span></div></div>
    <div class="stat"><div class="stat-label">Wind Speed</div><div class="stat-value" id="wind">—<span class="stat-unit">km/h</span></div></div>
    <div class="stat"><div class="stat-label">Yesterday's Rain</div><div class="stat-value" id="rainYest">—<span class="stat-unit">mm</span></div></div>
    <div class="stat"><div class="stat-label">Last Updated</div><div class="stat-value" style="font-size:0.9rem" id="updated">—</div></div>
  </div>

  <div class="chart-card">
    <div class="chart-label">Next 5 hours &mdash; confidence forecast</div>
    <div class="chart-wrap">
      <div class="midline"></div>
      <div class="bars" id="forecastBars"></div>
    </div>
    <div class="bar-labels" id="forecastLabels"></div>
  </div>

  <div class="footer-row">
    <div><span class="dot live" id="liveDot"></span>updates every 10 min &mdash; GET /api/hourly</div>
    <button class="refresh-btn" onclick="fetchHourly()">&#x21BB; refresh</button>
  </div>
</div>
<script>
function updateClock() {
  const h = String(new Date().getHours()).padStart(2,'0');
  document.getElementById('clockLabel').textContent = h + ':00';
}
updateClock(); setInterval(updateClock, 60000);

async function fetchHourly() {
  try {
    const d = await fetch('/api/hourly').then(r=>r.json());
    if (!d.timestamp) return;

    // --- Verdict card ---
    const card = document.getElementById('verdictCard');
    card.className = 'verdict-card ' + (d.will_rain ? 'rain' : 'dry');
    document.getElementById('verdictText').textContent = d.will_rain ? 'RAIN' : 'NO RAIN';
    document.getElementById('confBar').style.width = d.confidence + '%';
    document.getElementById('confPct').textContent  = d.confidence + '%';
    document.getElementById('rawOutput').textContent = 'raw model output: ' + d.raw_output;

    // --- Conditions ---
    const w = d.weather || {};
    document.getElementById('temp').innerHTML     = (w.temperature    ?? '—') + '<span class="stat-unit">°C</span>';
    document.getElementById('humidity').innerHTML = (w.humidity       ?? '—') + '<span class="stat-unit">%</span>';
    document.getElementById('pressure').innerHTML = (w.pressure       ?? '—') + '<span class="stat-unit">hPa</span>';
    document.getElementById('wind').innerHTML     = (w.wind_speed     ?? '—') + '<span class="stat-unit">km/h</span>';
    document.getElementById('rainYest').innerHTML = (w.rain_yesterday ?? '—') + '<span class="stat-unit">mm</span>';
    const ts = new Date(d.timestamp);
    document.getElementById('updated').textContent = ts.toLocaleTimeString('en-AU',{hour:'2-digit',minute:'2-digit'});

    // --- Forecast bar chart ---
    const barsEl   = document.getElementById('forecastBars');
    const labelsEl = document.getElementById('forecastLabels');
    barsEl.innerHTML = '';
    labelsEl.innerHTML = '';
    const HALF_PX = 60; // max px each bar can grow (half chart height)

    (d.forecast || []).forEach((h, i) => {
      const col = document.createElement('div');
      col.className = 'bar-col' + (i === 0 ? ' current' : '');

      const barH = Math.max(4, h.confidence / 100 * HALF_PX);

      if (h.will_rain) {
        const bar = document.createElement('div');
        bar.className = 'rain-bar';
        bar.style.height = barH + 'px';
        col.appendChild(bar);
        if (barH > 14) {
          const lbl = document.createElement('div');
          lbl.className = 'bar-conf';
          lbl.style.bottom = 'calc(50% + ' + (barH - 16) + 'px)';
          lbl.textContent = h.confidence + '%';
          col.appendChild(lbl);
        }
      } else {
        const bar = document.createElement('div');
        bar.className = 'dry-bar';
        bar.style.height = barH + 'px';
        col.appendChild(bar);
        if (barH > 14) {
          const lbl = document.createElement('div');
          lbl.className = 'bar-conf';
          lbl.style.top = 'calc(50% + ' + (barH - 16) + 'px)';
          lbl.textContent = h.confidence + '%';
          col.appendChild(lbl);
        }
      }

      barsEl.appendChild(col);

      // Hour label row
      const timeLbl = document.createElement('div');
      timeLbl.className = 'bar-time' + (i === 0 ? ' current' : '');
      timeLbl.textContent = h.label;
      labelsEl.appendChild(timeLbl);
    });

    document.getElementById('liveDot').className = 'dot live';
  } catch(e) {
    console.error(e);
    document.getElementById('liveDot').className = 'dot';
  }
}
fetchHourly(); setInterval(fetchHourly, 30000);
</script>
</body></html>"""


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    print("Loading models...")
    d_weights, d_biases = load_daily_model()
    d_ns                = load_daily_norm()
    h_model, h_ns, h_device = load_hourly_model()

    print("Running initial cycles...")
    run_all_cycles(d_weights, d_biases, d_ns, h_model, h_ns, h_device)

    t = threading.Thread(
        target=polling_thread,
        args=(d_weights, d_biases, d_ns, h_model, h_ns, h_device),
        daemon=True
    )
    t.start()

    PORT   = 8765
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"\nServer running at http://localhost:{PORT}")
    print(f"  Daily:  http://localhost:{PORT}/")
    print(f"  Hourly: http://localhost:{PORT}/hourly")
    print(f"  API:    http://localhost:{PORT}/api/latest")
    print(f"          http://localhost:{PORT}/api/hourly")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")