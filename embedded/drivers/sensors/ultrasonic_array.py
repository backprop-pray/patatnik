import time

from .ultrasonic_hcsr04 import HCSR04


class DualUltrasonicArray:
    def __init__(
        self,
        sensor1_trig=23,
        sensor1_echo=24,
        sensor2_trig=27,
        sensor2_echo=17,
        sensor3_trig=5,
        sensor3_echo=6,
        settle_seconds=2.0,
        inter_sensor_delay_seconds=0.005,
    ):
        self._sensor1 = HCSR04(trig_pin=sensor1_trig, echo_pin=sensor1_echo, settle_seconds=settle_seconds)
        self._sensor2 = HCSR04(trig_pin=sensor2_trig, echo_pin=sensor2_echo, settle_seconds=settle_seconds)
        self._sensor3 = HCSR04(trig_pin=sensor3_trig, echo_pin=sensor3_echo, settle_seconds=settle_seconds)
        self.inter_sensor_delay_seconds = inter_sensor_delay_seconds

    def _normalize_sensor_id(self, sensor_id):
        value = str(sensor_id).strip().lower()
        if value in ('1', 's1', 'sensor1', 'left'):
            return 1
        if value in ('2', 's2', 'sensor2', 'right'):
            return 2
        if value in ('3', 's3', 'sensor3', 'middle', 'center'):
            return 3
        raise ValueError(f'Invalid sensor id: {sensor_id}')

    def read_sensor(self, sensor_id, timeout_seconds=0.03):
        sid = self._normalize_sensor_id(sensor_id)
        if sid == 1:
            return self._sensor1.read_distance_cm(timeout_seconds=timeout_seconds)
        if sid == 2:
            return self._sensor2.read_distance_cm(timeout_seconds=timeout_seconds)
        return self._sensor3.read_distance_cm(timeout_seconds=timeout_seconds)

    def read_all(self, timeout_seconds=0.03):
        d1 = self._sensor1.read_distance_cm(timeout_seconds=timeout_seconds)
        time.sleep(self.inter_sensor_delay_seconds)
        d2 = self._sensor2.read_distance_cm(timeout_seconds=timeout_seconds)
        time.sleep(self.inter_sensor_delay_seconds)
        d3 = self._sensor3.read_distance_cm(timeout_seconds=timeout_seconds)
        return {1: d1, 2: d2, 3: d3}

    def cleanup(self):
        self._sensor1.cleanup()
        self._sensor2.cleanup()
        self._sensor3.cleanup()
