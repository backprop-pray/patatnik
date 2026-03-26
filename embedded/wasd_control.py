#!/usr/bin/env python3
import sys
import termios
import tty

from api import RoverAPI


def read_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main():
    rover = RoverAPI()
    print('WASD motor control')
    print('w: forward | s: backward | a: spin left | d: spin right | x: stop | q: quit')

    try:
        while True:
            key = read_key().lower()

            if key == 'w':
                rover.drive('forward', 'forward')
                print('forward')
            elif key == 's':
                rover.drive('backward', 'backward')
                print('backward')
            elif key == 'a':
                rover.drive('backward', 'forward')
                print('spin left')
            elif key == 'd':
                rover.drive('forward', 'backward')
                print('spin right')
            elif key == 'x':
                rover.stop_motors()
                print('stop')
            elif key == 'q':
                print('quit')
                break
    finally:
        rover.stop_motors()
        rover.close()
        print('motors stopped and GPIO cleaned up')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
