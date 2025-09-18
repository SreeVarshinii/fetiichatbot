import os
import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://fetii:fetii@db:5432/fetii")
engine = create_engine(DB_URL, future=True)

FETII_XLSX = os.getenv("FETII_XLSX", "/data/FetiiAI_Data_Austin.xlsx")
TRIPS_CSV   = os.getenv("FETII_TRIPS_CSV")     # optional override
RIDERS_CSV  = os.getenv("FETII_RIDERS_CSV")
DEMO_CSV    = os.getenv("FETII_DEMO_CSV")

def pick_sheet(xls, candidates):
    for name in xls.sheet_names:
        ln = name.lower().strip()
        if any(k in ln for k in candidates):
            return name
    return xls.sheet_names[0]

def normalize_trips(df):
    m = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ["trip id","tripid","trip_id"]: m[c] = "trip_id"
        elif lc in ["user id of booker","user_id_booker","booker","booker_user_id"]: m[c] = "user_id_booker"
        elif "pickup" in lc and "address" in lc: m[c] = "pickup_address"
        elif ("drop" in lc or "dropoff" in lc) and "address" in lc: m[c] = "dropoff_address"
        elif "pickup" in lc and "lat" in lc: m[c] = "pickup_lat"
        elif "pickup" in lc and ("lon" in lc or "long" in lc): m[c] = "pickup_lon"
        elif "drop" in lc and "lat" in lc: m[c] = "dropoff_lat"
        elif "drop" in lc and ("lon" in lc or "long" in lc): m[c] = "dropoff_lon"
        elif "timestamp" in lc or "pickup time" in lc: m[c] = "pickup_ts"
        elif "drop" in lc and "time" in lc: m[c] = "dropoff_ts"
        elif lc in ["ridercount","num riders","number of riders","passenger count"]: m[c] = "rider_count"
    return df.rename(columns=m)

def load_frames():
    if TRIPS_CSV and RIDERS_CSV and DEMO_CSV:
        trips  = pd.read_csv(TRIPS_CSV)
        riders = pd.read_csv(RIDERS_CSV)
        demo   = pd.read_csv(DEMO_CSV)
    else:
        xls = pd.ExcelFile(FETII_XLSX)
        trip_sheet  = pick_sheet(xls, ["trip"])
        rider_sheet = pick_sheet(xls, ["rider","passenger"])
        demo_sheet  = pick_sheet(xls, ["demo","age"])
        trips  = pd.read_excel(xls, sheet_name=trip_sheet)
        riders = pd.read_excel(xls, sheet_name=rider_sheet)
        demo   = pd.read_excel(xls, sheet_name=demo_sheet)

    trips = normalize_trips(trips)

    # riders: map to trip_id, user_id
    def map_cols(df, trip_keys, user_keys):
        lower = {c.lower().strip(): c for c in df.columns}
        tcol = next((lower[k] for k in trip_keys if k in lower), df.columns[0])
        ucol = next((lower[k] for k in user_keys if k in lower), df.columns[1])
        return df.rename(columns={tcol:"trip_id", ucol:"user_id"})

    riders = map_cols(riders, {"trip id","tripid","trip_id"}, {"user id","userid","user_id"})

    # demo: user_id, age
    lower = {c.lower().strip(): c for c in demo.columns}
    ucol = next((lower[k] for k in ["user id","userid","user_id"] if k in lower), demo.columns[0])
    acol = next((c for c in demo.columns if "age" in c.lower()), demo.columns[-1])
    demo = demo.rename(columns={ucol:"user_id", acol:"age"})

    return trips, riders, demo

def write_tables(trips, riders, demo):
    trips.to_sql("trips", engine, if_exists="replace", index=False)
    riders.to_sql("riders", engine, if_exists="replace", index=False)
    demo.to_sql("ride_demo", engine, if_exists="replace", index=False)
    with engine.begin() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_riders_trip ON riders(trip_id);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_demo_user ON ride_demo(user_id);"))

if __name__ == "__main__":
    t, r, d = load_frames()
    write_tables(t, r, d)
    print("Loaded tables: trips, riders, ride_demo")
