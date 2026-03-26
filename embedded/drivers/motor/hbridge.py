import RPi.GPIO as GPIO


class DualHBridgeMotorDriver:
    def __init__(self, m1_in1=20, m1_in2=21, m2_in1=16, m2_in2=12):
        self.m1_in1 = m1_in1
        self.m1_in2 = m1_in2
        self.m2_in1 = m2_in1
        self.m2_in2 = m2_in2
        self.pins = (self.m1_in1, self.m1_in2, self.m2_in1, self.m2_in2)

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in self.pins:
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

    def set_states(self, m1_in1, m1_in2, m2_in1, m2_in2):
        GPIO.output(self.m1_in1, GPIO.HIGH if m1_in1 else GPIO.LOW)
        GPIO.output(self.m1_in2, GPIO.HIGH if m1_in2 else GPIO.LOW)
        GPIO.output(self.m2_in1, GPIO.HIGH if m2_in1 else GPIO.LOW)
        GPIO.output(self.m2_in2, GPIO.HIGH if m2_in2 else GPIO.LOW)

    def set_both_forward(self):
        self.set_states(1, 0, 1, 0)

    def set_both_reverse(self):
        self.set_states(0, 1, 0, 1)

    def stop(self):
        self.set_states(0, 0, 0, 0)

    def cleanup(self):
        self.stop()
        GPIO.cleanup()
