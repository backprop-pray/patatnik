import time

import RPi.GPIO as GPIO


class HCSR04:
    def __init__(self, trig_pin=23, echo_pin=24, settle_seconds=2.0):
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.trig_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.echo_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

        GPIO.output(self.trig_pin, False)
        time.sleep(settle_seconds)

    def read_distance_cm(self, timeout_seconds=0.03):
        GPIO.output(self.trig_pin, True)
        time.sleep(0.00001)
        GPIO.output(self.trig_pin, False)

        start_deadline = time.time() + timeout_seconds
        pulse_start = time.time()
        while GPIO.input(self.echo_pin) == 0:
            pulse_start = time.time()
            if pulse_start > start_deadline:
                return None

        end_deadline = time.time() + timeout_seconds
        pulse_end = time.time()
        while GPIO.input(self.echo_pin) == 1:
            pulse_end = time.time()
            if pulse_end > end_deadline:
                return None

        pulse_duration = pulse_end - pulse_start
        distance_cm = pulse_duration * 17150.0
        return round(distance_cm, 2)

    def cleanup(self):
        GPIO.cleanup((self.trig_pin, self.echo_pin))
