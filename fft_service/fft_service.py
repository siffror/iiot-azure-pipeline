# fft_service.py
import json, time, os, collections, threading
import numpy as np
import paho.mqtt.client as mqtt
from http.server import BaseHTTPRequestHandler, HTTPServer

HOST = os.getenv("MQTT_HOST", "mosquitto")
PORT = int(os.getenv("MQTT_PORT", "1883"))
RATE = float(os.getenv("RATE", "200"))     # Hz
SPAN = float(os.getenv("SPAN", "2.0"))     # sekunder per FFT
TOP_IN = os.getenv("TOPIC_IN", "sensors/vibration")
TOP_OUT = os.getenv("TOPIC_OUT", "sensors/vibration_fft")
HEALTH_PORT = int(os.getenv("HEALTH_PORT", "5000"))

# --- minimal HTTP /health för Azure Container Apps probes ---
class _Health(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200); self.end_headers(); self.wfile.write(b"OK")
        else:
            self.send_response(404); self.end_headers()
def _run_health():
    HTTPServer(("0.0.0.0", HEALTH_PORT), _Health).serve_forever()
threading.Thread(target=_run_health, daemon=True).start()
# ------------------------------------------------------------

buf = collections.defaultdict(lambda: {"ax": [], "ay": []})
cli = mqtt.Client()
cli.connect(HOST, PORT, 60)
cli.loop_start()

def publish_fft(sid, axis, data, ts):
    if len(data) < 8:
        return
    x = np.asarray(data[-int(RATE * SPAN):], dtype=float)
    x = x - np.mean(x)
    win = np.hanning(len(x))
    X = np.fft.rfft(x * win)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / RATE)
    amp = np.abs(X) / (len(x) / 2.0)
    N = 20  # skicka topp N bin (minskar skrivvolym)
    idx = np.argsort(amp)[-N:][::-1]
    now_ms = ts
    for i in idx:
        payload = {
            "sensor_id": sid,
            "axis": axis,
            "bin_hz": int(freqs[i]),
            "amp": float(amp[i]),
            "ts": now_ms
        }
        cli.publish(f"{TOP_OUT}/{axis}", json.dumps(payload), qos=0, retain=False)

def on_msg(_c, _u, msg):
    try:
        x = json.loads(msg.payload.decode())
        sid = x.get("sensor_id", "imu01")
        ts = x.get("ts", int(time.time() * 1000))
        ax = float(x.get("ax", 0.0))
        ay = float(x.get("ay", 0.0))
        b = buf[sid]
        b["ax"].append(ax); b["ay"].append(ay)
        if len(b["ax"]) >= int(RATE * SPAN):
            publish_fft(sid, "ax", b["ax"], ts)
            publish_fft(sid, "ay", b["ay"], ts)
            b["ax"].clear(); b["ay"].clear()
    except Exception as e:
        # håll processen vid liv även om ett meddelande är fel
        print("on_msg error:", e, flush=True)

sub = mqtt.Client()
sub.connect(HOST, PORT, 60)
sub.subscribe(TOP_IN, qos=0)
sub.on_message = on_msg

print("▶ fft_service lyssnar på", TOP_IN, "→ skickar", TOP_OUT + "/ax", ",", TOP_OUT + "/ay", flush=True)
sub.loop_forever()
