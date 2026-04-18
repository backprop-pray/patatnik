#!/usr/bin/env python3
import argparse
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Picamera2 quick hardware test')
    parser.add_argument('--output', default='/home/yasen/patatnik/embedded/picam2_test.jpg')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--warmup', type=float, default=1.5)
    parser.add_argument('--duration', type=float, default=3.0, help='FPS probe duration in seconds')
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from picamera2 import Picamera2
    except Exception as exc:
        print(f'Picamera2 import failed: {exc}')
        print('Install with: sudo apt install -y python3-picamera2')
        return 2

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    picam2 = Picamera2()
    try:
        config = picam2.create_preview_configuration(
            main={'size': (args.width, args.height), 'format': 'RGB888'}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(max(0.0, args.warmup))

        picam2.capture_file(str(output))
        print(f'Captured test image: {output}')

        frames = 0
        start = time.monotonic()
        while time.monotonic() - start < max(0.1, args.duration):
            picam2.capture_array('main')
            frames += 1
        elapsed = time.monotonic() - start
        fps = frames / elapsed if elapsed > 0 else 0.0
        print(f'Frame probe: {frames} frames in {elapsed:.2f}s (~{fps:.2f} FPS)')
    finally:
        try:
            picam2.stop()
        except Exception:
            pass
        try:
            picam2.close()
        except Exception:
            pass

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
