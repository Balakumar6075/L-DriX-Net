import time
from collections import deque


class FatigueProgressionTracker:

    def __init__(self):

        # ============================================================
        # FATIGUE SCORE
        # ============================================================

        self.fatigue = 0.0

        self.minimum_score = 0.0
        self.maximum_score = 100.0

        self.last_update = None

        # ============================================================
        # CONCENTRATION TRACKING
        # ============================================================

        self.focused_start = None

        # Fatigue does not start decreasing immediately.
        #
        # Driver must remain continuously concentrated for
        # this amount of time before recovery begins.
        #
        # 5 minutes
        self.recovery_start_seconds = 300.0

        # ============================================================
        # EYE CLOSURE TRACKING
        # ============================================================

        self.eye_closed = False
        self.eye_closed_start = None

        # ============================================================
        # BLINK HISTORY
        # ============================================================

        self.blink_events = deque()

        # ============================================================
        # FATIGUE HISTORY
        # ============================================================

        self.score_history = deque(maxlen=30)

        # ============================================================
        # EYE PARAMETERS
        # ============================================================

        # Normal blink duration
        self.normal_blink_max_seconds = 0.8

        # Eye closure longer than this adds fatigue
        self.fatigue_closure_seconds = 1.0

        # Strong prolonged closure
        self.strong_closure_seconds = 2.0

        # Blink counting window
        self.blink_window_seconds = 60.0

        # ============================================================
        # FATIGUE INCREASE
        # ============================================================

        # Gradual fatigue increase.
        #
        # 0.015 points / second
        #
        # Approximately:
        #
        # 0.9 points / minute
        # 4.5 points / 5 minutes
        # 9 points / 10 minutes
        # 27 points / 30 minutes
        # 54 points / 1 hour
        #
        # This gives a visible gradual progression without
        # immediately reaching 100.
        self.base_fatigue_rate = 0.015

        # ============================================================
        # CONCENTRATION RECOVERY
        # ============================================================

        # Very slow recovery after sustained concentration.
        #
        # 0.005 points / second
        # 0.3 points / minute
        #
        # Once recovery starts, fatigue gradually comes down.
        self.concentration_recovery_rate = 0.005

        # ============================================================
        # LONG EYE CLOSURE PENALTIES
        # ============================================================

        self.long_closure_base = 0.5

        self.long_closure_rate = 0.8

        self.strong_closure_penalty = 1.0

        # ============================================================
        # HEAD POSE
        # ============================================================

        self.head_fatigue_rate = 0.003

    # ================================================================
    # CLAMP
    # ================================================================

    def _clamp(self, value):

        return max(
            self.minimum_score,
            min(
                self.maximum_score,
                value
            )
        )

    # ================================================================
    # PROCESS EYE STATE
    # ================================================================

    def _process_eye_state(
        self,
        eyes_closed,
        now
    ):

        # ------------------------------------------------------------
        # EYES JUST CLOSED
        # ------------------------------------------------------------

        if (
            eyes_closed
            and
            not self.eye_closed
        ):

            self.eye_closed = True

            self.eye_closed_start = now

        # ------------------------------------------------------------
        # EYES JUST OPENED
        # ------------------------------------------------------------

        elif (
            not eyes_closed
            and
            self.eye_closed
        ):

            if self.eye_closed_start is not None:

                duration = (
                    now -
                    self.eye_closed_start
                )

                # ====================================================
                # BLINK
                # ====================================================

                # Keep normal blink detection.
                #
                # Only closures between 0.08 and 0.8 seconds
                # are counted as blinks.

                if (
                    duration >= 0.08
                    and
                    duration <=
                    self.normal_blink_max_seconds
                ):

                    self.blink_events.append(now)

                # ====================================================
                # LONG EYE CLOSURE
                # ====================================================

                if (
                    duration >
                    self.fatigue_closure_seconds
                ):

                    excess = (
                        duration -
                        self.fatigue_closure_seconds
                    )

                    penalty = (
                        self.long_closure_base
                        +
                        min(
                            excess *
                            self.long_closure_rate,
                            4.0
                        )
                    )

                    self.fatigue += penalty

                # ====================================================
                # VERY LONG EYE CLOSURE
                # ====================================================

                if (
                    duration >
                    self.strong_closure_seconds
                ):

                    self.fatigue += (
                        self.strong_closure_penalty
                    )

            self.eye_closed = False

            self.eye_closed_start = None

    # ================================================================
    # CALCULATE BLINK COUNT
    # ================================================================

    def _calculate_blink_rate(self, now):

        while (
            self.blink_events
            and
            (
                now -
                self.blink_events[0]
            )
            >
            self.blink_window_seconds
        ):

            self.blink_events.popleft()

        return len(
            self.blink_events
        )

    # ================================================================
    # UPDATE
    # ================================================================

    def update(
        self,
        eyes_closed=False,
        head_pose=None,
        gaze=None,
        now=None
    ):

        # ============================================================
        # CURRENT TIME
        # ============================================================

        if now is None:

            now = time.monotonic()

        # ============================================================
        # FIRST UPDATE
        # ============================================================

        if self.last_update is None:

            self.last_update = now

        # ============================================================
        # ELAPSED TIME
        # ============================================================

        elapsed = max(
            0.0,
            now -
            self.last_update
        )

        self.last_update = now

        # ============================================================
        # DEFAULT INPUTS
        # ============================================================

        head_pose = head_pose or {}

        gaze = gaze or {}

        # ============================================================
        # HEAD POSE
        # ============================================================

        yaw = float(
            head_pose.get(
                "yaw",
                0.0
            )
        )

        pitch = float(
            head_pose.get(
                "pitch",
                0.0
            )
        )

        # ============================================================
        # PROCESS EYE STATE
        # ============================================================

        self._process_eye_state(
            eyes_closed,
            now
        )

        # ============================================================
        # BLINK COUNT
        # ============================================================

        blink_count = (
            self._calculate_blink_rate(
                now
            )
        )

        # ============================================================
        # DETERMINE CONCENTRATION
        # ============================================================

        focused = (
            abs(yaw) < 12
            and
            abs(pitch) < 10
            and
            not eyes_closed
        )

        # ============================================================
        # TRACK FOCUSED DURATION
        # ============================================================

        if focused:

            if self.focused_start is None:

                self.focused_start = now

            focused_duration = (
                now -
                self.focused_start
            )

        else:

            # Any loss of concentration resets
            # the continuous concentration timer.

            self.focused_start = None

            focused_duration = 0.0

        # ============================================================
        # FATIGUE PROGRESSION
        # ============================================================

        if focused:

            # --------------------------------------------------------
            # DRIVER IS CONCENTRATED
            # --------------------------------------------------------
            #
            # IMPORTANT:
            #
            # Fatigue DOES NOT increase while concentrated.
            #
            # This is what you requested.
            #

            pass

        else:

            # --------------------------------------------------------
            # DRIVER IS NOT CONCENTRATED
            # --------------------------------------------------------
            #
            # Gradually increase fatigue.
            #

            self.fatigue += (
                self.base_fatigue_rate
                *
                elapsed
            )

        # ============================================================
        # LONG-TERM CONCENTRATION RECOVERY
        # ============================================================

        if (
            focused
            and
            focused_duration >=
            self.recovery_start_seconds
        ):

            # --------------------------------------------------------
            # Driver has been concentrated for 5+ minutes.
            #
            # Now fatigue gradually decreases.
            # --------------------------------------------------------

            recovery = (
                self.concentration_recovery_rate
                *
                elapsed
            )

            self.fatigue -= recovery

        # ============================================================
        # RAPID BLINKING
        # ============================================================

        # Keep blink contribution small.
        #
        # Blink count itself is NOT modified here.

        blink_fatigue = 0.0

        if blink_count > 30:

            blink_fatigue = 0.002

        elif blink_count > 24:

            blink_fatigue = 0.0012

        elif blink_count > 18:

            blink_fatigue = 0.0005

        self.fatigue += (
            blink_fatigue
            *
            elapsed
        )

        # ============================================================
        # HEAD POSITION FATIGUE
        # ============================================================

        head_fatigue = 0.0

        if pitch > 15:

            head_fatigue = 0.006

        elif pitch > 10:

            head_fatigue = 0.002

        self.fatigue += (
            head_fatigue
            *
            elapsed
        )

        # ============================================================
        # CURRENT EYE CLOSURE
        # ============================================================

        current_closure_duration = 0.0

        if (
            self.eye_closed
            and
            self.eye_closed_start is not None
        ):

            current_closure_duration = (
                now -
                self.eye_closed_start
            )

            # --------------------------------------------------------
            # Continuous eye closure > 1 second
            # --------------------------------------------------------

            if (
                current_closure_duration >
                1.0
            ):

                self.fatigue += (
                    0.008
                    *
                    elapsed
                )

        # ============================================================
        # ONE-HOUR CONCENTRATION RESET
        # ============================================================

        if (
            focused
            and
            focused_duration >= 3600
        ):

            self.fatigue = 0.0

        # ============================================================
        # LIMIT FATIGUE TO 0-100
        # ============================================================

        self.fatigue = self._clamp(
            self.fatigue
        )

        # ============================================================
        # STORE HISTORY
        # ============================================================

        self.score_history.append(
            self.fatigue
        )

        # ============================================================
        # TREND
        # ============================================================

        trend = "STABLE"

        if (
            len(self.score_history)
            >= 5
        ):

            old_score = (
                self.score_history[-5]
            )

            new_score = (
                self.score_history[-1]
            )

            difference = (
                new_score -
                old_score
            )

            if difference > 0.5:

                trend = "INCREASING"

            elif difference < -0.5:

                trend = "DECREASING"

        # ============================================================
        # RETURN RESULT
        # ============================================================

        return {

            "fatigue_index": round(
                self.fatigue,
                2
            ),

            "trend": trend,

            "eyes_closed": bool(
                self.eye_closed
            ),

            "eye_closure_duration": round(
                current_closure_duration,
                2
            ),

            "blink_count_60s": blink_count,

            "focused_duration": round(
                focused_duration,
                1
            ),

            "status": self._status()
        }

    # ================================================================
    # STATUS
    # ================================================================

    def _status(self):

        if self.fatigue < 20:

            return "LOW"

        elif self.fatigue < 40:

            return "NORMAL"

        elif self.fatigue < 60:

            return "ELEVATED"

        elif self.fatigue < 80:

            return "HIGH"

        else:

            return "VERY HIGH"

    # ================================================================
    # RESET
    # ================================================================

    def reset(self):

        self.fatigue = 0.0

        self.last_update = None

        self.focused_start = None

        self.eye_closed = False

        self.eye_closed_start = None

        self.blink_events.clear()

        self.score_history.clear()