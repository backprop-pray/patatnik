#!/usr/bin/env python3
import sys
import termios
import tty

from api import RoverAPI

SPEED_STEP = 10
DEFAULT_SPEED = 60


def clamp_speed(value):
    return max(0, min(100, int(value)))


def read_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def print_help():
    print('WASD motor control with speed', flush=True)
    print('w: forward | s: backward | a: spin left | d: spin right', flush=True)
    print('x: stop | q: quit', flush=True)
    print('+: speed up | -: speed down', flush=True)
    print('1-9: set speed 10-90 | 0: set speed 100', flush=True)


def main():
    rover = RoverAPI()
    speed = DEFAULT_SPEED
    left_direction = 'stop'
    right_direction = 'stop'
    print_help()
    print(f'Current speed: {speed}%', flush=True)

    try:
        while True:
            key = read_key()

            if key in ('+', '='):
                speed = clamp_speed(speed + SPEED_STEP)
                if left_direction != 'stop' or right_direction != 'stop':
                    rover.drive(left_direction, right_direction, left_speed=speed, right_speed=speed)
                    print(f'speed: {speed}% (applied)', flush=True)
                else:
                    print(f'speed: {speed}%', flush=True)
                continue

            if key == '-':
                speed = clamp_speed(speed - SPEED_STEP)
                if left_direction != 'stop' or right_direction != 'stop':
                    rover.drive(left_direction, right_direction, left_speed=speed, right_speed=speed)
                    print(f'speed: {speed}% (applied)', flush=True)
                else:
                    print(f'speed: {speed}%', flush=True)
                continue

            if key.isdigit():
                speed = 100 if key == '0' else int(key) * 10
                if left_direction != 'stop' or right_direction != 'stop':
                    rover.drive(left_direction, right_direction, left_speed=speed, right_speed=speed)
                    print(f'speed: {speed}% (applied)', flush=True)
                else:
                    print(f'speed: {speed}%', flush=True)
                continue

            key = key.lower()
            if key == 'w':
                left_direction, right_direction = 'forward', 'forward'
                rover.drive(left_direction, right_direction, left_speed=speed, right_speed=speed)
                print(f'forward @ {speed}%', flush=True)
            elif key == 's':
                left_direction, right_direction = 'backward', 'backward'
                rover.drive(left_direction, right_direction, left_speed=speed, right_speed=speed)
                print(f'backward @ {speed}%', flush=True)
            elif key == 'a':
                left_direction, right_direction = 'backward', 'forward'
                rover.drive(left_direction, right_direction, left_speed=speed, right_speed=speed)
                print(f'spin left @ {speed}%', flush=True)
            elif key == 'd':
                left_direction, right_direction = 'forward', 'backward'
                rover.drive(left_direction, right_direction, left_speed=speed, right_speed=speed)
                print(f'spin right @ {speed}%', flush=True)
            elif key == 'x':
                left_direction, right_direction = 'stop', 'stop'
                rover.stop_motors()
                print('stop', flush=True)
            elif key == 'q':
                print('quit', flush=True)
                break
    finally:
        rover.stop_motors()
        rover.close()
        print('motors stopped and GPIO cleaned up', flush=True)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
