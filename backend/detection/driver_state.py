import numpy as np


class DriverStateDetector:
    """
    Temporal driver-state detector for the L-DriX-Net prototype.

    States:
        CONCENTRATED
        DISTRACTED
        DROWSY

    This is a prototype heuristic detector.

    It uses:
        - head pose
        - gaze position
        - gaze movement
        - temporal behavior

    A temporal window is used instead of making a decision
    from a single frame.

    IMPORTANT:
        The scores produced by this class are heuristic
        confidence-like scores. They are NOT probabilities
        from a trained neural-network classifier.
    """

    STATES = [
        "CONCENTRATED",
        "DISTRACTED",
        "DROWSY",
    ]

    def __init__(self, window_size=16):
        self.window_size = window_size
        self.history = []

    # =========================================================
    # SAFE NUMERIC HELPERS
    # =========================================================

    @staticmethod
    def _safe_float(value, default=0.0):
        """
        Safely convert a value to float.

        Invalid, missing, or non-finite values are replaced
        with the supplied default.
        """

        try:
            value = float(value)

            if not np.isfinite(value):
                return default

            return value

        except (TypeError, ValueError):
            return default

    @staticmethod
    def _mean(values):
        """
        Calculate the mean of a list safely.
        """

        if not values:
            return 0.0

        return float(np.mean(values))

    @staticmethod
    def _std(values):
        """
        Calculate standard deviation safely.
        """

        if not values:
            return 0.0

        return float(np.std(values))

    @staticmethod
    def _range(values):
        """
        Calculate max - min safely.
        """

        if not values:
            return 0.0

        return float(np.max(values) - np.min(values))

    # =========================================================
    # TEMPORAL FEATURE EXTRACTION
    # =========================================================

    def _calculate_features(self, history):
        """
        Calculate temporal behavioral features from the
        current history window.

        The history contains measurements from multiple
        consecutive frames.
        """

        yaw = []
        pitch = []
        roll = []

        gaze_x = []
        gaze_y = []

        # -----------------------------------------------------
        # Collect frame-level measurements
        # -----------------------------------------------------

        for item in history:

            head = item.get("head_pose", {})
            gaze = item.get("gaze", {})

            # Head pose
            yaw.append(
                self._safe_float(
                    head.get("yaw", 0.0)
                )
            )

            pitch.append(
                self._safe_float(
                    head.get("pitch", 0.0)
                )
            )

            roll.append(
                self._safe_float(
                    head.get("roll", 0.0)
                )
            )

            # Gaze position
            location = gaze.get(
                "gaze_location_2d",
                [0.0, 0.0]
            )

            if isinstance(location, (list, tuple)) and len(location) >= 2:

                gaze_x.append(
                    self._safe_float(
                        location[0]
                    )
                )

                gaze_y.append(
                    self._safe_float(
                        location[1]
                    )
                )

        # -----------------------------------------------------
        # Gaze movement between consecutive frames
        # -----------------------------------------------------

        gaze_dx = []
        gaze_dy = []

        for i in range(1, len(gaze_x)):

            gaze_dx.append(
                abs(
                    gaze_x[i] -
                    gaze_x[i - 1]
                )
            )

            gaze_dy.append(
                abs(
                    gaze_y[i] -
                    gaze_y[i - 1]
                )
            )

        # -----------------------------------------------------
        # Return temporal feature dictionary
        # -----------------------------------------------------

        return {

            # Head pose mean
            "yaw_mean":
                self._mean(yaw),

            "pitch_mean":
                self._mean(pitch),

            "roll_mean":
                self._mean(roll),

            # Head pose variation
            "yaw_std":
                self._std(yaw),

            "pitch_std":
                self._std(pitch),

            "roll_std":
                self._std(roll),

            # Head pose range
            "yaw_range":
                self._range(yaw),

            "pitch_range":
                self._range(pitch),

            "roll_range":
                self._range(roll),

            # Gaze position
            "gaze_x_mean":
                self._mean(gaze_x),

            "gaze_y_mean":
                self._mean(gaze_y),

            # Gaze variation
            "gaze_x_std":
                self._std(gaze_x),

            "gaze_y_std":
                self._std(gaze_y),

            # Gaze range
            "gaze_x_range":
                self._range(gaze_x),

            "gaze_y_range":
                self._range(gaze_y),

            # Gaze movement
            "gaze_movement":
                self._mean(gaze_dx)
                +
                self._mean(gaze_dy),

            "gaze_x_movement":
                self._mean(gaze_dx),

            "gaze_y_movement":
                self._mean(gaze_dy),
        }

    # =========================================================
    # STATE SCORING
    # =========================================================

    def _calculate_scores(self, features):
        """
        Calculate prototype driver-state scores.

        States:
            CONCENTRATED
            DISTRACTED
            DROWSY

        The scores are heuristic confidence-like values.

        They are NOT probabilities produced by a trained
        neural-network classifier.
        """

        # =====================================================
        # DROWSINESS
        # =====================================================

        drowsy_score = 0.0

        # -----------------------------------------------------
        # Primary drowsiness signal:
        # sustained downward head posture.
        #
        # We deliberately require pitch evidence.
        # Low gaze movement by itself is NOT sufficient.
        # -----------------------------------------------------

        if features["pitch_mean"] > 18:

            drowsy_score += 0.60

        elif features["pitch_mean"] > 12:

            drowsy_score += 0.40

        elif features["pitch_mean"] > 9:

            drowsy_score += 0.20

        # -----------------------------------------------------
        # Stable downward posture provides additional evidence.
        # -----------------------------------------------------

        if (
            features["pitch_mean"] > 9
            and
            features["pitch_std"] < 3
        ):

            drowsy_score += 0.15

        # -----------------------------------------------------
        # Low gaze activity is supporting evidence ONLY when
        # the head is already sufficiently downward.
        # -----------------------------------------------------

        if features["pitch_mean"] > 9:

            if features["gaze_x_std"] < 20:

                drowsy_score += 0.10

            if features["gaze_y_std"] < 12:

                drowsy_score += 0.10

        # -----------------------------------------------------
        # Strong sideways head orientation should be classified
        # as distraction rather than drowsiness.
        # -----------------------------------------------------

        if abs(features["yaw_mean"]) > 15:

            drowsy_score *= 0.10

        drowsy_score = min(
            drowsy_score,
            1.0
        )

        # =====================================================
        # DISTRACTION
        # =====================================================

        distraction_score = 0.0

        # -----------------------------------------------------
        # Primary distraction signal:
        # sustained sideways head orientation.
        # -----------------------------------------------------

        if abs(features["yaw_mean"]) > 25:

            distraction_score += 0.55

        elif abs(features["yaw_mean"]) > 18:

            distraction_score += 0.40

        elif abs(features["yaw_mean"]) > 12:

            distraction_score += 0.25

        # -----------------------------------------------------
        # Downward head posture can also indicate attention
        # away from the road.
        # -----------------------------------------------------

        if abs(features["pitch_mean"]) > 15:

            distraction_score += 0.20

        elif abs(features["pitch_mean"]) > 10:

            distraction_score += 0.10

        # -----------------------------------------------------
        # Large horizontal gaze variation can indicate
        # attention shifts.
        # -----------------------------------------------------

        if features["gaze_x_std"] > 100:

            distraction_score += 0.20

        elif features["gaze_x_std"] > 70:

            distraction_score += 0.10

        # -----------------------------------------------------
        # Large temporal head movements provide additional
        # evidence of attention shifts.
        # -----------------------------------------------------

        if features["yaw_range"] > 30:

            distraction_score += 0.10

        if features["pitch_range"] > 20:

            distraction_score += 0.10

        distraction_score = min(
            distraction_score,
            1.0
        )

        # =====================================================
        # CONCENTRATION
        # =====================================================

        concentration_score = 0.0

        # -----------------------------------------------------
        # Driver is generally facing forward.
        # -----------------------------------------------------

        if abs(features["yaw_mean"]) < 12:

            concentration_score += 0.35

        # -----------------------------------------------------
        # Head is not significantly downward/upward.
        # -----------------------------------------------------

        if abs(features["pitch_mean"]) < 10:

            concentration_score += 0.30

        # -----------------------------------------------------
        # Stable head orientation.
        # -----------------------------------------------------

        if features["yaw_std"] < 8:

            concentration_score += 0.15

        if features["pitch_std"] < 6:

            concentration_score += 0.10

        # -----------------------------------------------------
        # Some natural gaze movement is compatible with
        # normal driving/scanning.
        # -----------------------------------------------------

        if (
            10
            <= features["gaze_x_std"]
            <= 100
        ):

            concentration_score += 0.10

        concentration_score = min(
            concentration_score,
            1.0
        )

        # =====================================================
        # MUTUAL EXCLUSION
        # =====================================================

        # -----------------------------------------------------
        # Strong sideways orientation should strongly suppress
        # concentration.
        # -----------------------------------------------------

        if abs(features["yaw_mean"]) > 18:

            concentration_score *= 0.25

        # -----------------------------------------------------
        # Strong downward posture should suppress concentration.
        # -----------------------------------------------------

        if features["pitch_mean"] > 15:

            concentration_score *= 0.25

        # -----------------------------------------------------
        # Strong drowsiness should suppress the other states.
        # -----------------------------------------------------

        if drowsy_score >= 0.70:

            concentration_score *= 0.25

            distraction_score *= 0.50

        # -----------------------------------------------------
        # Strong distraction should suppress concentration.
        # -----------------------------------------------------

        if distraction_score >= 0.50:

            concentration_score *= 0.25

        # =====================================================
        # FINAL SCORE DICTIONARY
        # =====================================================

        return {

            "CONCENTRATED":
                concentration_score,

            "DISTRACTED":
                distraction_score,

            "DROWSY":
                drowsy_score,
        }

    # =========================================================
    # TEMPORAL UPDATE
    # =========================================================

    def update(self, frame_data):
        """
        Add a new frame to the temporal history and calculate
        the current driver state.

        The detector waits until the temporal window is full
        before producing a state prediction.
        """

        # -----------------------------------------------------
        # Add current frame
        # -----------------------------------------------------

        self.history.append(
            frame_data
        )

        # -----------------------------------------------------
        # Keep only the most recent frames
        # -----------------------------------------------------

        if len(self.history) > self.window_size:

            self.history.pop(0)

        # -----------------------------------------------------
        # Temporal calibration
        # -----------------------------------------------------

        if len(self.history) < self.window_size:

            return {

                "state":
                    "CALIBRATING",

                "confidence":
                    0.0,

                "scores": {

                    "CONCENTRATED":
                        0.0,

                    "DISTRACTED":
                        0.0,

                    "DROWSY":
                        0.0,
                },

                "frames":
                    len(self.history),

                "required_frames":
                    self.window_size,
            }

        # -----------------------------------------------------
        # Calculate temporal features
        # -----------------------------------------------------

        features = self._calculate_features(
            self.history
        )

        # -----------------------------------------------------
        # Calculate state scores
        # -----------------------------------------------------

        scores = self._calculate_scores(
            features
        )

        # -----------------------------------------------------
        # Select state with highest score
        # -----------------------------------------------------

        state = max(
            scores,
            key=scores.get
        )

        confidence = scores[state]

        # -----------------------------------------------------
        # Return complete result
        # -----------------------------------------------------

        return {

            "state":
                state,

            "confidence":
                round(
                    confidence,
                    4
                ),

            "scores": {

                key:
                    round(
                        value,
                        4
                    )

                for key, value
                in scores.items()
            },

            "frames":
                len(self.history),

            "required_frames":
                self.window_size,

            "features":
                features,
        }

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):
        """
        Clear the temporal history.

        This should be called when:
            - a new driver starts
            - the camera changes
            - the live session ends
            - a new video begins
        """

        self.history = []


# =============================================================
# DIRECT MODULE TEST
# =============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("DriverStateDetector module initialized successfully.")
    print("=" * 70)

    detector = DriverStateDetector(
        window_size=16
    )

    print(
        "Temporal window:",
        detector.window_size
    )

    print(
        "States:",
        detector.STATES
    )