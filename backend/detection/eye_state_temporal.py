import time
from collections import deque


class TemporalEyeStateProcessor:

    def __init__(
        self,
        history_size=5,
        closed_threshold=0.60,
        blink_min_seconds=0.08,
        blink_max_seconds=0.80,
        prolonged_closure_seconds=1.0
    ):

        self.history_size = history_size

        self.closed_threshold = (
            closed_threshold
        )

        self.blink_min_seconds = (
            blink_min_seconds
        )

        self.blink_max_seconds = (
            blink_max_seconds
        )

        self.prolonged_closure_seconds = (
            prolonged_closure_seconds
        )

        self.state_history = deque(
            maxlen=history_size
        )

        self.current_state = "UNKNOWN"

        self.previous_state = "UNKNOWN"

        self.eye_closed = False

        self.closure_start = None

        self.total_blinks = 0

        self.last_blink_time = None

        self.prolonged_closure = False

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        left_result,
        right_result,
        now=None
    ):

        if now is None:

            now = time.monotonic()

        left_closed_probability = float(
            left_result.get(
                "closed_probability",
                0.0
            )
        )

        right_closed_probability = float(
            right_result.get(
                "closed_probability",
                0.0
            )
        )

        average_closed_probability = (
            left_closed_probability
            +
            right_closed_probability
        ) / 2.0

        # ----------------------------------------------------
        # RAW STATE
        # ----------------------------------------------------

        raw_closed = (
            average_closed_probability
            >= self.closed_threshold
        )

        raw_state = (
            "CLOSED"
            if raw_closed
            else "OPEN"
        )

        self.state_history.append(
            raw_state
        )

        # ----------------------------------------------------
        # STABLE STATE
        # ----------------------------------------------------

        stable_state = (
            self._get_stable_state()
        )

        self.previous_state = (
            self.current_state
        )

        self.current_state = (
            stable_state
        )

        # ----------------------------------------------------
        # PROCESS EYE STATE
        # ----------------------------------------------------

        self._process_transition(
            raw_state,
            now
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        return self._build_result(
            left_result,
            right_result,
            average_closed_probability,
            now
        )

    # ========================================================
    # STABLE STATE
    # ========================================================

    def _get_stable_state(self):

        if (
            len(self.state_history)
            < self.history_size
        ):

            return "UNKNOWN"

        closed_count = sum(
            state == "CLOSED"
            for state in self.state_history
        )

        open_count = sum(
            state == "OPEN"
            for state in self.state_history
        )

        if closed_count > open_count:

            return "CLOSED"

        return "OPEN"

    # ========================================================
    # TRANSITION
    # ========================================================

    def _process_transition(
        self,
        state,
        now
    ):

        # ----------------------------------------------------
        # CLOSED
        # ----------------------------------------------------

        if state == "CLOSED":

            if not self.eye_closed:

                self.eye_closed = True

                self.closure_start = now

                self.prolonged_closure = False

            return

        # ----------------------------------------------------
        # OPEN
        # ----------------------------------------------------

        if state == "OPEN":

            if self.eye_closed:

                if self.closure_start is not None:

                    duration = (
                        now
                        - self.closure_start
                    )

                    # ----------------------------------------
                    # BLINK
                    # ----------------------------------------

                    if (
                        self.blink_min_seconds
                        <= duration
                        <= self.blink_max_seconds
                    ):

                        self.total_blinks += 1

                        self.last_blink_time = (
                            now
                        )

                self.eye_closed = False

                self.closure_start = None

                self.prolonged_closure = False

    # ========================================================
    # RESULT
    # ========================================================

    def _build_result(
        self,
        left_result,
        right_result,
        average_closed_probability,
        now
    ):

        closure_duration = 0.0

        if (
            self.eye_closed
            and
            self.closure_start is not None
        ):

            closure_duration = (
                now
                - self.closure_start
            )

            self.prolonged_closure = (
                closure_duration
                >=
                self.prolonged_closure_seconds
            )

        return {

            "state":
                self.current_state,

            "previous_state":
                self.previous_state,

            "eyes_closed":
                self.current_state
                == "CLOSED",

            "left_state":
                left_result.get(
                    "state",
                    "UNKNOWN"
                ),

            "right_state":
                right_result.get(
                    "state",
                    "UNKNOWN"
                ),

            "left_closed_probability":
                round(
                    float(
                        left_result.get(
                            "closed_probability",
                            0.0
                        )
                    ),
                    4
                ),

            "right_closed_probability":
                round(
                    float(
                        right_result.get(
                            "closed_probability",
                            0.0
                        )
                    ),
                    4
                ),

            "average_closed_probability":
                round(
                    average_closed_probability,
                    4
                ),

            "closure_duration":
                round(
                    closure_duration,
                    2
                ),

            "prolonged_closure":
                bool(
                    self.prolonged_closure
                ),

            "total_blinks":
                self.total_blinks,

            "last_blink_time":
                self.last_blink_time,

            "history_size":
                len(
                    self.state_history
                )
        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.state_history.clear()

        self.current_state = "UNKNOWN"

        self.previous_state = "UNKNOWN"

        self.eye_closed = False

        self.closure_start = None

        self.total_blinks = 0

        self.last_blink_time = None

        self.prolonged_closure = False


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print(
        "L-DriX-Net Temporal Eye-State Processor"
    )
    print("=" * 60)

    processor = (
        TemporalEyeStateProcessor()
    )

    print(
        "Processor initialized successfully."
    )

    print("=" * 60)