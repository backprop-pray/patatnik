#!/usr/bin/env python3
import sys
import time
from pathlib import Path

from drivers.gps.nmea import extract_sentences, parse_lat_lon
from drivers.gps.uart import UartReader

GPS_PORT = '/dev/ttyAMA0'
GPS_BAUD = 9600
STATUS_INTERVAL = 5.0
FALLBACK_FILE = '/home/yasen/gps_fallback.env'


def load_fallback_coords(file_path=FALLBACK_FILE):
    path = Path(file_path)
    if not path.exists():
        return None

    values = {}
    for raw_line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        values[key.strip()] = value.strip()

    lat_raw = values.get('GPS_FALLBACK_LAT')
    lon_raw = values.get('GPS_FALLBACK_LON')
    if not lat_raw or not lon_raw:
        return None

    try:
        return float(lat_raw), float(lon_raw)
    except ValueError:
        return None


def print_fallback(coords):
    lat, lon = coords
    print(f'{lat:.6f},{lon:.6f} (FALLBACK)')
    sys.stdout.flush()


def main():
    fallback = load_fallback_coords()
    reader = UartReader(port=GPS_PORT, baud=GPS_BAUD)

    try:
        reader.open()
    except OSError as exc:
        print(f'Failed to open {GPS_PORT}: {exc}', file=sys.stderr)
        if fallback is not None:
            print_fallback(fallback)
        else:
            print('NO_FIX')
            sys.stdout.flush()
        return 1

    print(
        f'Reading GPS on {GPS_PORT} at {GPS_BAUD} baud. Waiting for fix...',
        file=sys.stderr,
    )

    last_status = 0.0

    try:
        for line in reader.iter_lines():
            got_fix = False
            for sentence in extract_sentences(line):
                latlon = parse_lat_lon(sentence)
                if latlon is None:
                    continue
                lat, lon = latlon
                print(f'{lat:.6f},{lon:.6f}')
                sys.stdout.flush()
                got_fix = True

            now = time.monotonic()
            if not got_fix and now - last_status >= STATUS_INTERVAL:
                if fallback is not None:
                    print('No GPS fix yet. Using configured fallback coordinates.', file=sys.stderr)
                    print_fallback(fallback)
                else:
                    print('No GPS fix yet.', file=sys.stderr)
                    print('NO_FIX')
                    sys.stdout.flush()
                last_status = now
    except KeyboardInterrupt:
        print('Stopped.', file=sys.stderr)
        return 0
    finally:
        reader.close()


if __name__ == '__main__':
    raise SystemExit(main())
