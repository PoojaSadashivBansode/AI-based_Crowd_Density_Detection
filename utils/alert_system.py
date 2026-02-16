import datetime

class AlertSystem:
    def __init__(self, threshold=50):
        """
        Initialize Alert System.
        Args:
            threshold (int): Count threshold to trigger alert.
        """
        self.threshold = threshold
        self.last_alert_time = None
        self.alert_cooldown = 5 # Seconds

    def check_alert(self, count):
        """
        Check if count exceeds threshold and trigger alert.
        Args:
            count (int): Current crowd count.
        Returns:
            bool: True if alert triggered, False otherwise.
            str: Alert message.
        """
        if count >= self.threshold:
            current_time = datetime.datetime.now()
            if self.last_alert_time is None or (current_time - self.last_alert_time).total_seconds() > self.alert_cooldown:
                self.last_alert_time = current_time
                message = f"ALERT: Crowd count {count} exceeds threshold {self.threshold}!"
                print(f"\033[91m{message}\033[0m") # Red text
                # Here you can add sound or email logic
                return True, message
        return False, None
