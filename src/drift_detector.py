from river import drift
import numpy as np


class FraudDriftDetector:
    """
    Drift detector for fraud-imbalanced streams.

    Two complementary signals are tracked:
    1. Overall error rate (ADWIN, sensitive delta) — catches broad degradation.
    2. Fraud-recall proxy: errors on samples where the model gave any fraud
       probability (prob > FRAUD_PROB_THRESHOLD). On a 99.9% non-fraud stream
       the overall error rate is dominated by correct non-fraud predictions and
       barely moves even when all fraud is missed. Watching only the fraud-
       suspicious subset makes the signal ~100x stronger.
    """

    FRAUD_PROB_THRESHOLD = 0.3  # sample considered "fraud-suspicious"

    def __init__(self, delta=0.05):
        # Lower delta = more sensitive (was 0.002, which was far too conservative
        # for a stream where overall error rate never exceeds ~0.5%)
        self.overall_detector = drift.ADWIN(delta=delta)
        self.fraud_detector = drift.ADWIN(delta=delta)
        self.drift_detected = False
        self.n_drifts = 0

    def update(self, error_value, fraud_prob=None):
        """
        Update the detector.

        Args:
            error_value: 1 if prediction wrong, 0 if correct
            fraud_prob:  model's predicted fraud probability (float, optional).
                        When provided, feeds the fraud-suspicious channel.
        """
        self.overall_detector.update(error_value)
        if self.overall_detector.drift_detected:
            self.drift_detected = True
            self.n_drifts += 1

        if fraud_prob is not None and fraud_prob > self.FRAUD_PROB_THRESHOLD:
            self.fraud_detector.update(error_value)
            if self.fraud_detector.drift_detected:
                self.drift_detected = True
                self.n_drifts += 1

    def reset(self):
        self.drift_detected = False


if __name__ == "__main__":
    detector = FraudDriftDetector()
    for i in range(100):
        error = 0 if i < 50 else 1
        detector.update(error, fraud_prob=0.4 if i > 40 else 0.1)
        if detector.drift_detected:
            print(f"Drift detected at step {i}")
            detector.reset()
    print(f"Total drifts: {detector.n_drifts}")
