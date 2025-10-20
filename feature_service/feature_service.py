
import numpy as np
from influxdb_client import InfluxDBClient, Point, WriteOptions
import os, time

INFLUX_URL = os.getenv("INFLUX_URL", "http://influxdb:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG", "iiot")
BUCKET_RAW = os.getenv("BUCKET_RAW", "machine_health")
BUCKET_FEATURES = os.getenv("BUCKET_FEATURES", "features")

client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
query_api = client.query_api()
write_api = client.write_api(write_options=WriteOptions(batch_size=1))

def compute_features(data):
    arr = np.array(data)
    rms = np.sqrt(np.mean(arr ** 2))
    kurt = np.mean((arr - np.mean(arr))**4) / (np.var(arr)**2)
    return {"rms": rms, "kurtosis": kurt}

def main():
    print("Feature service started.")
    while True:
        query = f'from(bucket:"{BUCKET_RAW}") |> range(start: -1m)'
        tables = query_api.query(query)
        for table in tables:
            values = [record.get_value() for record in table.records]
            if not values: continue
            f = compute_features(values)
            point = Point("features").field("rms", f["rms"]).field("kurtosis", f["kurtosis"])
            write_api.write(bucket=BUCKET_FEATURES, record=point)
        time.sleep(10)

if __name__ == "__main__":
    main()
