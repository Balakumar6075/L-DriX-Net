from backend.detection.driver_state import DriverStateDetector


detector = DriverStateDetector(
    window_size=16
)


print("=" * 70)
print("L-DriX-Net Driver State Detector Test")
print("=" * 70)


# ---------------------------------------------------------
# TEST 1: CONCENTRATED
# ---------------------------------------------------------

print("\nCONCENTRATED TEST")
print("-" * 70)

detector.reset()

result = None

for i in range(16):

    result = detector.update({

        "head_pose": {
            "yaw": 4.0,
            "pitch": 3.0,
            "roll": 0.0,
        },

        "gaze": {
            "gaze_location_2d": [
                470 + i * 2,
                250 + i
            ]
        }

    })

print(result)


# ---------------------------------------------------------
# TEST 2: DISTRACTED
# ---------------------------------------------------------

print("\nDISTRACTED TEST")
print("-" * 70)

detector.reset()

result = None

for i in range(16):

    result = detector.update({

        "head_pose": {
            "yaw": 25.0,
            "pitch": 2.0,
            "roll": 0.0,
        },

        "gaze": {
            "gaze_location_2d": [
                700 + i * 5,
                250
            ]
        }

    })

print(result)


# ---------------------------------------------------------
# TEST 3: DROWSY
# ---------------------------------------------------------

print("\nDROWSY TEST")
print("-" * 70)

detector.reset()

result = None

for i in range(16):

    result = detector.update({

        "head_pose": {
            "yaw": 2.0,
            "pitch": 18.0,
            "roll": 0.0,
        },

        "gaze": {
            "gaze_location_2d": [
                470 + (i % 2),
                450 + (i % 2)
            ]
        }

    })

print(result)


print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)