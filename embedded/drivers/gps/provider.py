import os
import select
import time
from pathlib import Path

from .nmea import extract_sentences, parse_lat_lon
from .uart import UartReader


class GPSProvider:
    def __init__(self, port='/dev/ttyAMA0', baud=9600, fallback_file='/home/yasen/gps_fallback.env'):
        self.port = port
        self.baud = baud
        self.fallback_file = fallback_file
        self._reader = UartReader(port=self.port, baud=self.baud)
        self._is_open = False
        self._buffer = ''

    def _load_fallback_coords(self):
        path = Path(self.fallback_file)
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

    def open(self):
        if self._is_open:
            return
        self._reader.open()
        self._is_open = True

    def close(self):
        if self._is_open:
            self._reader.close()
            self._is_open = False

    def get_position(self, timeout_seconds=2.0, max_sentences=40, allow_fallback=True):
        try:
            self.open()
        except OSError:
            if allow_fallback:
                coords = self._load_fallback_coords()
                if coords is not None:
                    lat, lon = coords
                    return {'lat': lat, 'lon': lon, 'source': 'fallback', 'fix': False}
            return {'lat': None, 'lon': None, 'source': 'none', 'fix': False}

        deadline = time.monotonic() + timeout_seconds
        seen = 0

        while time.monotonic() < deadline and seen < max_sentences:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            ready, _, _ = select.select([self._reader.fd], [], [], remaining)
            if not ready:
                break

            chunk = os.read(self._reader.fd, 1024)
            if not chunk:
                continue

            self._buffer += chunk.decode('ascii', errors='ignore')

            while '\n' in self._buffer:
                raw_line, self._buffer = self._buffer.split('\n', 1)
                line = raw_line.strip()
                if not line:
                    continue

                for sentence in extract_sentences(line):
                    latlon = parse_lat_lon(sentence)
                    seen += 1
                    if latlon is not None:
                        lat, lon = latlon
                        return {'lat': lat, 'lon': lon, 'source': 'gps', 'fix': True}
                    if seen >= max_sentences:
                        break

        if allow_fallback:
            coords = self._load_fallback_coords()
            if coords is not None:
                lat, lon = coords
                return {'lat': lat, 'lon': lon, 'source': 'fallback', 'fix': False}

        return {'lat': None, 'lon': None, 'source': 'none', 'fix': False}
