import os
import time
import numpy as np
from influxdb_client import InfluxDBClient, Point, WriteOptions

# Environment variables (inject via ACA secrets/env)
INFLUX_URL = os.getenv("INFLUX_URL", "http://influxdb:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG", "iiot")
BUCKET_RAW = os.getenv("BUCKET_RAW", "machine_health")
BUCKET_FEATURES = os.getenv("BUCKET_FEATURES", "features")

# Connect to Influx
client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
query_api = client.query_api()
write_api = client.write_api(write_options=WriteOptions(batch_size=1))

def compute_features(data):
    """Compute RMS and Kurtosis from sensor data array"""
    arr = np.array(data)
    if len(arr) == 0:
        return None
    rms = np.sqrt(np.mean(arr ** 2))
    kurt = np.mean((arr - np.mean(arr)) ** 4) / (np.var(arr) ** 2)
    return {"rms": float(rms), "kurtosis": float(kurt)}

def main():
    print("🚀 Feature service started")
    while True:
        query = f'from(bucket:"{BUCKET_RAW}") |> range(start: -1m)'
        tables = query_api.query(query)
        for table in tables:
            values = [record.get_value() for record in table.records if isinstance(record.get_value(), (int, float))]
            if not values:
                continue
            features = compute_features(values)
            if features:
                point = Point("features") \
                    .field("rms", features["rms"]) \
                    .field("kurtosis", features["kurtosis"])
                write_api.write(bucket=BUCKET_FEATURES, record=point)
                print(f"✅ Wrote features: {features}")
        time.sleep(10)

if __name__ == "__main__":
    main()
