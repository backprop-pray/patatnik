#!/usr/bin/env python3
import time

from api import RoverAPI

FORWARD_1_SECONDS = 2.0
SPIN_SECONDS = 1.5
FORWARD_2_SECONDS = 2.0


def main():
    rover = RoverAPI()

    try:
        rover.drive('forward', 'forward')
        print('Step 1: both motors forward')
        time.sleep(FORWARD_1_SECONDS)

        rover.drive('backward', 'forward')
        print('Step 2: spin (left motor backward, right motor forward)')
        time.sleep(SPIN_SECONDS)

        rover.drive('forward', 'forward')
        print('Step 3: both motors forward again')
        time.sleep(FORWARD_2_SECONDS)
    finally:
        rover.stop_motors()
        rover.close()
        print('Done: motors stopped and GPIO cleaned up.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
