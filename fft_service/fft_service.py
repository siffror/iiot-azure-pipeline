import os, json, time, signal, sys, traceback, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point, WriteOptions

# -----------------------------
# Environment (from ACA)
# -----------------------------
MQTT_HOST = os.getenv("MQTT_HOST", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC_IN = os.getenv("MQTT_TOPIC_IN", "iiot/vibration_raw")
MQTT_TOPIC_OUT = os.getenv("MQTT_TOPIC_OUT", "iiot/vibration_fft")
SAMPLE_RATE = float(os.getenv("SAMPLE_RATE", "25600"))  # Hz

INFLUX_URL = os.getenv("INFLUXDB_URL", "http://influxdb:8086")
INFLUX_TOKEN = os.getenv("INFLUXDB_TOKEN", "")
INFLUX_ORG = os.getenv("INFLUXDB_ORG", "iiot-org")
INFLUX_BUCKET = os.getenv("INFLUXDB_BUCKET", "iiot")

MODEL_PATH = os.getenv("MODEL_PATH", "")  # e.g. /models/iforest_final.joblib

# -----------------------------
# Tiny /health HTTP server (port 5000) for ACA probes
# -----------------------------
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404); self.end_headers()
    def log_message(self, *_):  # no noisy logs
        pass

def start_health_server():
    server = HTTPServer(("0.0.0.0", 5000), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server

# -----------------------------
# Optional model loading
# -----------------------------
model = None
if MODEL_PATH:
    try:
        import joblib
        model = joblib.load(MODEL_PATH)
        print(f"[fft] Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[fft] WARN: Could not load model from {MODEL_PATH}: {e}", file=sys.stderr)
        traceback.print_exc()

# -----------------------------
# Influx client (optional)
# -----------------------------
influx = None
write_api = None
if INFLUX_TOKEN:
    try:
        influx = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        write_api = influx.write_api(write_options=WriteOptions(batch_size=500, flush_interval=1000))
        print("[fft] InfluxDB client initialized")
    except Exception as e:
        print(f"[fft] WARN: Influx init failed: {e}", file=sys.stderr)

# -----------------------------
# FFT
# -----------------------------
def compute_fft(signal_np, fs):
    n = len(signal_np)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    spec = np.abs(np.fft.rfft(signal_np))
    return freqs, spec

# -----------------------------
# MQTT callbacks
# -----------------------------
def on_connect(client, userdata, flags, rc):
    print(f"[fft] MQTT connected rc={rc}; subscribing {MQTT_TOPIC_IN}")
    client.subscribe(MQTT_TOPIC_IN, qos=0)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        # Expected: {"time": 1699999999999, "ax":[...], "ay":[...]}  (axes optional)
        ts = int(payload.get("time", int(time.time() * 1000)))

        for axis in ("ax", "ay"):
            if axis not in payload:
                continue

            x = np.asarray(payload[axis], dtype=float)
            if x.ndim != 1:
                raise ValueError(f"{axis} array must be 1-D")

            f, s = compute_fft(x, SAMPLE_RATE)
            dom_idx = int(np.argmax(s))
            dom_freq = float(f[dom_idx])
            energy = float(np.sum(s ** 2))

            out = {
                "time": ts,
                "axis": axis,
                "freq": f.tolist(),
                "spec": s.tolist(),
                "dominant_freq": dom_freq,
                "energy": energy
            }

            # Optional anomaly score if model is loaded (IsolationForest etc.)
            if model is not None:
                try:
                    # simple features from spectrum (can be improved later)
                    features = np.array([[dom_freq, energy]], dtype=float)
                    if hasattr(model, "score_samples"):
                        score = float(model.score_samples(features)[0])
                    elif hasattr(model, "decision_function"):
                        score = float(model.decision_function(features)[0])
                    else:
                        score = 0.0
                    out["anomaly_score"] = score
                except Exception as e:
                    print(f"[fft] WARN: model scoring failed: {e}", file=sys.stderr)

            topic = f"{MQTT_TOPIC_OUT}/{axis}"  # e.g. iiot/vibration_fft/ax
            client.publish(topic, json.dumps(out), qos=0, retain=False)

            # Optional write to Influx
            if write_api is not None:
                try:
                    p = Point("vibration_fft") \
                        .tag("axis", axis) \
                        .field("dominant_freq", dom_freq) \
                        .field("energy", energy) \
                        .time(ts * 1_000_000, write_precision="ns")
                    write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=p)
                except Exception as e:
                    print(f"[fft] WARN: Influx write failed: {e}", file=sys.stderr)

    except Exception as e:
        print(f"[fft] ERROR processing message: {e}", file=sys.stderr)
        traceback.print_exc()

def main():
    # Start health endpoint for probes
    _ = start_health_server()

    client = mqtt.Client(client_id="fft-service")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)

    def _shutdown(*_):
        print("[fft] shutdown")
        try:
            client.disconnect()
        except Exception:
            pass
        if influx:
            try: influx.close()
            except Exception: pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    client.loop_forever()

if __name__ == "__main__":
    main()
