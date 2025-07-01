import threading

class SensorCollector:
    def __init__(self):
        self.latest_imu = None
        self.latest_lux = None
        self.lock = threading.Lock()

    def on_imu_received(self, samples, imu_idx):
        with self.lock:
            self.latest_imu = samples[0]

    def on_baro_received(self, sample):
        # Replace with ALS if available in your SDK
        with self.lock:
            # Try to get lux, else fallback to pressure
            self.latest_lux = getattr(sample, 'lux', None) or getattr(sample, 'pressure', None)

    def get_latest(self):
        with self.lock:
            return self.latest_imu, self.latest_lux

def calc_depth(frame, imu_pkt):
    # Placeholder: real implementation would use frame and IMU to estimate depth
    # For now, return None
    return None

def build_prompt(image_path, lux=None, depth=None):
    context = []
    if lux is not None:
        if lux < 50:
            context.append("The scene is very dimly lit.")
        else:
            context.append("The scene is well-lit.")
    if depth is not None:
        context.append(f"The main subject is about {depth:.1f} metres away.")
    context_hint = " ".join(context) if context else None
    return context_hint 