import csv


CSV_FILE = "real_driver_state_analysis.csv"


print("=" * 70)
print("DROWSY WINDOW INSPECTION")
print("=" * 70)


rows = []

with open(
    CSV_FILE,
    "r",
    newline=""
) as file:

    reader = csv.DictReader(file)

    for row in reader:
        rows.append(row)


drowsy_rows = []

for row in rows:

    if row["state"] == "DROWSY":
        drowsy_rows.append(row)


print()
print("DROWSY WINDOWS FOUND:")
print(len(drowsy_rows))


print()
print("-" * 70)


for row in drowsy_rows:

    print(
        f"Frame {int(row['frame_id']):5d} | "
        f"Confidence {float(row['drowsy_score']):.2f} | "
        f"Yaw {float(row['yaw_mean']):7.2f}° | "
        f"Pitch {float(row['pitch_mean']):7.2f}° | "
        f"Yaw std {float(row['yaw_std']):6.2f} | "
        f"Pitch std {float(row['pitch_std']):6.2f} | "
        f"Gaze X std {float(row['gaze_x_std']):6.2f} | "
        f"Gaze Y std {float(row['gaze_y_std']):6.2f}"
    )


print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)