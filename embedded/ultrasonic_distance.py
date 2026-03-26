#!/usr/bin/env python3
import time

from drivers.sensors.ultrasonic_hcsr04 import HCSR04

SENSOR_1_TRIG_PIN = 23
SENSOR_1_ECHO_PIN = 24
SENSOR_2_TRIG_PIN = 27
SENSOR_2_ECHO_PIN = 17
SENSOR_3_TRIG_PIN = 13
SENSOR_3_ECHO_PIN = 6
READ_INTERVAL_SECONDS = 0.02
INTER_SENSOR_DELAY_SECONDS = 0.005


def format_distance(label, distance):
    if distance is None:
        return f'{label}: NO_ECHO'
    return f'{label}: {distance:.2f} cm'


def main():
    print('Distance measurement in progress (3 sensors)', flush=True)
    sensor_1 = HCSR04(trig_pin=SENSOR_1_TRIG_PIN, echo_pin=SENSOR_1_ECHO_PIN)
    sensor_2 = HCSR04(trig_pin=SENSOR_2_TRIG_PIN, echo_pin=SENSOR_2_ECHO_PIN)
    sensor_3 = HCSR04(trig_pin=SENSOR_3_TRIG_PIN, echo_pin=SENSOR_3_ECHO_PIN)

    try:
        while True:
            distance_1 = sensor_1.read_distance_cm()
            time.sleep(INTER_SENSOR_DELAY_SECONDS)
            distance_2 = sensor_2.read_distance_cm()
            time.sleep(INTER_SENSOR_DELAY_SECONDS)
            distance_3 = sensor_3.read_distance_cm()

            line = ' | '.join(
                [
                    format_distance('S1', distance_1),
                    format_distance('S2', distance_2),
                    format_distance('S3', distance_3),
                ]
            )
            print(line, flush=True)
            time.sleep(READ_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print('Stopped.', flush=True)
    finally:
        sensor_1.cleanup()
        sensor_2.cleanup()
        sensor_3.cleanup()


if __name__ == '__main__':
    main()
