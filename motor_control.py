#!/usr/bin/env python3
import time
import RPi.GPIO as GPIO

M1_IN1 = 20
M1_IN2 = 21
M2_IN1 = 16
M2_IN2 = 12
STEP_SECONDS = 2


def set_pair(high_pin, low_pin, label):
    GPIO.output(high_pin, GPIO.HIGH)
    GPIO.output(low_pin, GPIO.LOW)
    print(f"{label}: GPIO {high_pin}=HIGH, GPIO {low_pin}=LOW")


def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    pins = (M1_IN1, M1_IN2, M2_IN1, M2_IN2)
    for pin in pins:
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

    try:
        set_pair(M1_IN1, M1_IN2, "M1 forward")
        time.sleep(STEP_SECONDS)

        set_pair(M1_IN2, M1_IN1, "M1 reverse")
        time.sleep(STEP_SECONDS)

        set_pair(M2_IN1, M2_IN2, "M2 forward")
        time.sleep(STEP_SECONDS)

        set_pair(M2_IN2, M2_IN1, "M2 reverse")
        time.sleep(STEP_SECONDS)
    finally:
        for pin in pins:
            GPIO.output(pin, GPIO.LOW)
        GPIO.cleanup()
        print("Done. All pins set LOW and cleaned up.")


if __name__ == "__main__":
    main()
