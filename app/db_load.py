import os
import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://fetii:fetii@db:5432/fetii")
engine = create_engine(DB_URL, future=True)

# CSV paths (override with env vars)
TRIPS_CSV = os.getenv("FETII_TRIPS_CSV", "/data/Trip_data_extensive.csv")
USERS_CSV = os.getenv("FETII_USERS_CSV", "/data/User_extensive.csv")

def normalize_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Trip_data_extensive.csv into trips table schema."""
    m = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc == "trip id": m[c] = "trip_id"
        elif lc in ["booking user id"]: m[c] = "user_id_booker"
        elif "pick up latitude" in lc: m[c] = "pickup_lat"
        elif "pick up longitude" in lc: m[c] = "pickup_lon"
        elif "drop off latitude" in lc: m[c] = "dropoff_lat"
        elif "drop off longitude" in lc: m[c] = "dropoff_lon"
        elif "pick up address" in lc: m[c] = "pickup_address"
        elif "drop off address" in lc: m[c] = "dropoff_address"
        elif "trip date and time" in lc: m[c] = "pickup_ts"
        elif "total passengers" in lc: m[c] = "rider_count"
        elif lc == "pickup_place": m[c] = "pickup_place"
        elif lc == "pickup_street": m[c] = "pickup_street"
        elif lc == "drop_off_place": m[c] = "dropoff_place"
        elif lc == "drop_off_street": m[c] = "dropoff_street"
        elif lc == "date": m[c] = "date"
        elif lc == "time": m[c] = "time"
        elif lc == "day": m[c] = "day"

    out = df.rename(columns=m).copy()

    # Parse timestamp and derive date/time/day if missing
    if "pickup_ts" in out.columns:
        out["pickup_ts"] = pd.to_datetime(out["pickup_ts"], errors="coerce")
        if "date" not in out.columns:
            out["date"] = out["pickup_ts"].dt.date
        if "time" not in out.columns:
            out["time"] = out["pickup_ts"].dt.strftime("%H:%M:%S")
        if "day" not in out.columns:
            out["day"] = out["pickup_ts"].dt.day_name()

    # Convert numeric fields
    for col in ["rider_count", "pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out

def normalize_users(df: pd.DataFrame):
    """Split User_extensive.csv into riders and demo tables if possible."""
    lower = {c.lower().strip(): c for c in df.columns}
    tcol = next((lower[k] for k in ["trip id","tripid","trip_id"] if k in lower), None)
    ucol = next((lower[k] for k in ["user id","userid","user_id"] if k in lower), None)

    riders, demo = None, None

    if tcol and ucol:
        riders = df.rename(columns={tcol:"trip_id", ucol:"user_id"})[["trip_id","user_id"]].copy()

    # look for age or other demographics
    # agecol = next((c for c in df.columns if "age" in c.lower()), None)
    # if ucol and agecol:
    #     demo = df.rename(columns={ucol:"user_id", agecol:"age"})[["user_id","age"]].copy()
    #     demo["age"] = pd.to_numeric(demo["age"], errors="coerce")

    return riders

def load_frames():
    trips_raw = pd.read_csv(TRIPS_CSV)
    users_raw = pd.read_csv(USERS_CSV)

    trips = normalize_trips(trips_raw)
    riders = normalize_users(users_raw)

    return trips, riders

def write_tables(trips, riders):
    trips.to_sql("trips", engine, if_exists="replace", index=False)
    riders.to_sql("riders", engine, if_exists="replace", index=False)
    # if demo is not None:
    #     demo.to_sql("ride_demo", engine, if_exists="replace", index=False)

    with engine.begin() as conn:
        if riders is not None:
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_riders_trip ON riders(trip_id);"))
        # if demo is not None:
        #     conn.execute(text("CREATE INDEX IF NOT EXISTS idx_demo_user ON ride_demo(user_id);"))

if __name__ == "__main__":
    t, r = load_frames()
    write_tables(t, r)
    written = ["trips"]
    if r is not None: written.append("riders")
    # if d is not None: written.append("ride_demo")
    print(f"Loaded tables: {', '.join(written)}")
