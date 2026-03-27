#!/usr/bin/env python3
import argparse
import json
import logging
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from drivers.gps.provider import GPSProvider

try:
    import cv2
except Exception:
    cv2 = None

try:
    import numpy as np
except Exception:
    np = None


MAX_IMAGE_BYTES = 5 * 1024 * 1024


class LatestFrameBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._timestamp = 0.0

    def update(self, frame):
        with self._lock:
            self._frame = frame.copy()
            self._timestamp = time.time()

    def get(self):
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._timestamp


class OptionalUSBStreamClient:
    def __init__(self, laptop_ip, port, endpoint, enabled=True):
        self.laptop_ip = laptop_ip
        self.port = port
        self.endpoint = endpoint
        self._enabled = enabled

    @property
    def enabled(self):
        return self._enabled and bool(self.laptop_ip)

    def send_frame(self, frame, timestamp):
        if not self.enabled:
            return
        _ = frame, timestamp


class PlantReporter:
    def __init__(self, api_url, timeout_seconds=10.0, enabled=True):
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds
        self._enabled = enabled

    @property
    def enabled(self):
        return self._enabled and bool(self.api_url)

    @staticmethod
    def _encode_jpeg_under_limit(frame, max_bytes=MAX_IMAGE_BYTES):
        if cv2 is None or frame is None:
            return None

        scales = [1.0, 0.85, 0.7, 0.55]
        qualities = [90, 80, 70, 60, 50, 40]

        for scale in scales:
            candidate = frame
            if scale != 1.0:
                h, w = frame.shape[:2]
                nw = max(1, int(w * scale))
                nh = max(1, int(h * scale))
                candidate = cv2.resize(candidate, (nw, nh), interpolation=cv2.INTER_AREA)

            for quality in qualities:
                ok, encoded = cv2.imencode('.jpg', candidate, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                if not ok:
                    continue
                data = encoded.tobytes()
                if len(data) <= max_bytes:
                    return data

        return None

    @staticmethod
    def _build_multipart_body(latitude, longitude, image_bytes):
        boundary = f'----OpenCodeBoundary{uuid.uuid4().hex}'
        boundary_bytes = boundary.encode('utf-8')
        crlf = bytes((13, 10))

        chunks = []

        def add_text(name, value):
            chunks.append(b'--' + boundary_bytes + crlf)
            header = f'Content-Disposition: form-data; name="{name}"'.encode('utf-8')
            chunks.append(header + crlf + crlf)
            chunks.append(str(value).encode('utf-8') + crlf)

        add_text('latitude', latitude)
        add_text('longitude', longitude)

        chunks.append(b'--' + boundary_bytes + crlf)
        chunks.append(b'Content-Disposition: form-data; name="image"; filename="plant.jpg"' + crlf)
        chunks.append(b'Content-Type: image/jpeg' + crlf + crlf)
        chunks.append(image_bytes)
        chunks.append(crlf)
        chunks.append(b'--' + boundary_bytes + b'--' + crlf)

        body = b''.join(chunks)
        content_type = f'multipart/form-data; boundary={boundary}'
        return body, content_type

    def send(self, frame, gps_data, ai_result):
        if not self.enabled:
            logging.debug('Plant reporter disabled; skipping send.')
            return {'ok': False, 'reason': 'disabled'}

        lat = gps_data.get('lat') if gps_data else None
        lon = gps_data.get('lon') if gps_data else None
        if lat is None or lon is None:
            logging.warning('Anomaly detected but GPS coords are missing; skipping plant API send.')
            return {'ok': False, 'reason': 'missing_gps'}

        if cv2 is None:
            logging.warning('cv2 not available, cannot encode anomaly image.')
            return {'ok': False, 'reason': 'cv2_missing'}

        image_bytes = self._encode_jpeg_under_limit(frame, max_bytes=MAX_IMAGE_BYTES)
        if image_bytes is None:
            logging.warning('Could not encode image under 5MB limit for plant API.')
            return {'ok': False, 'reason': 'image_too_large_or_encode_failed'}

        body, content_type = self._build_multipart_body(lat, lon, image_bytes)
        req = urllib.request.Request(
            self.api_url,
            data=body,
            headers={'Content-Type': content_type},
            method='POST',
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                status_code = getattr(response, 'status', None)
                raw = response.read().decode('utf-8', errors='replace')
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = {'raw': raw}

            status = payload.get('status') if isinstance(payload, dict) else None
            message = payload.get('message') if isinstance(payload, dict) else None
            plant_id = None
            if isinstance(payload, dict) and isinstance(payload.get('data'), dict):
                plant_id = payload['data'].get('id')

            logging.info(
                'Plant API success http=%s status=%s id=%s message=%s',
                status_code,
                status,
                plant_id,
                message,
            )
            return {'ok': True, 'status_code': status_code, 'payload': payload, 'plant_id': plant_id}
        except urllib.error.HTTPError as exc:
            details = exc.read().decode('utf-8', errors='replace')
            logging.warning('Plant API HTTP error %s: %s', exc.code, details)
            return {'ok': False, 'reason': 'http_error', 'status_code': exc.code, 'details': details}
        except urllib.error.URLError as exc:
            logging.warning('Plant API unreachable: %s', exc)
            return {'ok': False, 'reason': 'url_error', 'details': str(exc)}
        except Exception as exc:
            logging.warning('Unexpected plant API send error: %s', exc)
            return {'ok': False, 'reason': 'unexpected', 'details': str(exc)}


def parse_args():
    parser = argparse.ArgumentParser(description='Orchestrator for PPO rover + optional USB AI monitor.')
    parser.add_argument('--ppo-script', default='ppo_rover.py', help='Path to ppo_rover.py')

    parser.add_argument('--disable-ppo', action='store_true', help='Disable PPO rover process.')
    parser.add_argument('--disable-usb', action='store_true', help='Disable USB camera capture loop.')
    parser.add_argument('--disable-ai', action='store_true', help='Disable AI monitor loop.')
    parser.add_argument('--disable-gps', action='store_true', help='Disable GPS reads in anomaly reporting.')
    parser.add_argument('--disable-usb-stream', action='store_true', help='Disable optional USB frame stream forwarding.')
    parser.add_argument('--disable-anomaly-report', action='store_true', help='Disable POST to plant API when anomaly is detected.')
    parser.add_argument('--disable-ppo-plant-gate', action='store_true', help='Do not wait for PPO plant-detected flag; run full pipeline on schedule.')

    parser.add_argument('--usb-device', default=None, help='USB camera device path or index, e.g. /dev/video8 or 1')
    parser.add_argument('--usb-width', type=int, default=1280)
    parser.add_argument('--usb-height', type=int, default=720)
    parser.add_argument('--usb-fps', type=int, default=30)
    parser.add_argument('--ai-interval', type=float, default=0.5, help='Seconds between AI checks.')
    parser.add_argument('--anomaly-cooldown', type=float, default=5.0, help='Min seconds between anomaly sends.')
    parser.add_argument('--force-anomaly', action='store_true', help='Force anomaly=true for testing send path.')

    parser.add_argument('--gps-port', default='/dev/ttyAMA0')
    parser.add_argument('--gps-baud', type=int, default=9600)
    parser.add_argument('--gps-fallback-file', default='/home/yasen/gps_fallback.env')

    parser.add_argument('--laptop-ip', default=None, help='Optional laptop IP for future USB stream forwarding.')
    parser.add_argument('--laptop-port', type=int, default=8080)
    parser.add_argument('--stream-endpoint', default='/stream/frame')

    parser.add_argument('--plant-api-url', default='https://morzio.com/api/plants')
    parser.add_argument('--plant-api-timeout', type=float, default=10.0)

    parser.add_argument('--ppo-plant-flag-file', default='/tmp/ppo_plant_detected.flag')
    parser.add_argument('--full-pipeline-cli', default='/home/yasen/patatnik/plant_pipeline/cli/full_pipeline_cli.py')
    parser.add_argument('--full-pipeline-timeout', type=float, default=120.0)
    parser.add_argument('--batch1-config', default=None)
    parser.add_argument('--batch2-config', default=None)
    parser.add_argument('--pipeline-frame-dir', default='/tmp/orchestrator_pipeline_frames')
    parser.add_argument('--mock-full-pipeline-disease', action='store_true', help='Mock full pipeline as disease detected (testing).')

    parser.add_argument('--test-plant-api', action='store_true', help='Send one mock multipart request then exit.')
    parser.add_argument('--mock-latitude', type=float, default=44.9521)
    parser.add_argument('--mock-longitude', type=float, default=34.1024)
    parser.add_argument('--mock-image-path', default=None, help='Optional image path for API test mode.')

    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    return parser.parse_args()


def build_usb_candidates(explicit_device):
    if explicit_device is not None:
        if str(explicit_device).isdigit():
            return [int(explicit_device)]
        return [str(explicit_device)]

    candidates = []
    by_id = Path('/dev/v4l/by-id')
    if by_id.exists():
        for dev in sorted(by_id.glob('usb-*-video-index0')):
            candidates.append(str(dev))
        for dev in sorted(by_id.glob('usb-*-video-index2')):
            candidates.append(str(dev))

    for dev in ('/dev/video8', '/dev/video10', '/dev/video0', '/dev/video2'):
        if Path(dev).exists():
            candidates.append(dev)

    unique = []
    seen = set()
    for item in candidates:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def try_open_camera(source, width, height, fps):
    if cv2 is None:
        return None

    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    for _ in range(5):
        ok, _ = cap.read()
        if ok:
            return cap

    cap.release()
    return None


def open_usb_camera(explicit_device, width, height, fps):
    candidates = build_usb_candidates(explicit_device)
    if not candidates:
        raise RuntimeError('No USB camera candidates found.')

    for source in candidates:
        cap = try_open_camera(source, width, height, fps)
        if cap is not None:
            return cap, source

    raise RuntimeError(f'Could not open any USB camera candidate: {candidates}')


def load_mock_frame(mock_image_path, width=1280, height=720):
    if cv2 is None:
        raise RuntimeError('cv2 is required for test mode image encoding.')

    if mock_image_path:
        image_path = Path(mock_image_path)
        if not image_path.exists():
            raise FileNotFoundError(f'Mock image not found: {image_path}')
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f'Could not read mock image: {image_path}')
        return frame

    if np is None:
        raise RuntimeError('numpy is required to generate mock frame when --mock-image-path is not provided.')

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (25, 60, 25)
    cv2.putText(frame, 'MOCK PLANT FRAME', (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    cv2.putText(frame, 'demo anomaly request', (40, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 255, 180), 2)
    cv2.putText(frame, time.strftime('%Y-%m-%d %H:%M:%S'), (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 255, 200), 2)
    return frame


def run_plant_api_test(args):
    logging.info('Running single plant API test request to %s', args.plant_api_url)
    frame = load_mock_frame(args.mock_image_path, width=args.usb_width, height=args.usb_height)
    gps = {
        'lat': args.mock_latitude,
        'lon': args.mock_longitude,
        'source': 'mock',
        'fix': False,
    }
    ai_result = {
        'is_sick': True,
        'is_anomaly': True,
        'label': 'mock_test',
        'score': 1.0,
    }

    reporter = PlantReporter(
        api_url=args.plant_api_url,
        timeout_seconds=args.plant_api_timeout,
        enabled=True,
    )
    result = reporter.send(frame=frame, gps_data=gps, ai_result=ai_result)
    if result.get('ok'):
        payload = result.get('payload')
        if isinstance(payload, dict):
            logging.info('Plant API test successful payload: %s', json.dumps(payload, ensure_ascii=True))
        return 0

    logging.error('Plant API test failed: %s', result)
    return 1


def ensure_flag_file(path):
    fp = Path(path)
    try:
        if not fp.exists():
            fp.write_text('0', encoding='utf-8')
    except Exception as exc:
        logging.warning('Could not initialize PPO flag file %s: %s', fp, exc)


def read_ppo_plant_flag(path):
    fp = Path(path)
    try:
        return int(fp.read_text(encoding='utf-8', errors='ignore').strip() or '0')
    except Exception:
        return 0


def clear_ppo_plant_flag(path):
    fp = Path(path)
    try:
        fp.write_text('0', encoding='utf-8')
    except Exception as exc:
        logging.warning('Could not clear PPO flag file %s: %s', fp, exc)


def extract_json_object(text):
    if not text:
        return None
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def detect_disease_from_full_pipeline(payload):
    if not isinstance(payload, dict):
        return False, None

    batch2 = payload.get('batch2')
    if not isinstance(batch2, dict):
        return False, None

    for key in ('disease', 'diagnosis', 'status', 'recommendedAction'):
        value = batch2.get(key)
        if value is None:
            continue
        text = str(value).strip().lower()
        if text and text not in ('none', 'null', 'normal', 'healthy', 'ok'):
            return True, str(value)

    label = str(batch2.get('label') or '').strip().lower()
    suspicious = bool(batch2.get('suspicious'))
    if suspicious or label in ('suspicious', 'uncertain', 'disease', 'diseased', 'anomaly', 'abnormal'):
        return True, label or 'suspicious'

    return False, label or None


def run_full_pipeline_on_frame(frame, args):
    if args.mock_full_pipeline_disease:
        payload = {
            'batch2': {
                'label': 'suspicious',
                'suspicious': True,
                'suspicious_score': 0.99,
            }
        }
        return {'ok': True, 'has_disease': True, 'label': 'suspicious', 'payload': payload}

    if cv2 is None:
        return {'ok': False, 'reason': 'cv2_missing'}

    script_path = Path(args.full_pipeline_cli).expanduser().resolve()
    if not script_path.exists():
        return {'ok': False, 'reason': f'full_pipeline_cli_not_found:{script_path}'}

    frame_dir = Path(args.pipeline_frame_dir).expanduser()
    frame_dir.mkdir(parents=True, exist_ok=True)
    image_path = frame_dir / f'pipeline_frame_{int(time.time() * 1000)}.jpg'

    if not cv2.imwrite(str(image_path), frame):
        return {'ok': False, 'reason': 'frame_write_failed'}

    cmd = [sys.executable, str(script_path), '--image', str(image_path)]
    if args.batch1_config:
        cmd.extend(['--batch1-config', args.batch1_config])
    if args.batch2_config:
        cmd.extend(['--batch2-config', args.batch2_config])

    cwd = script_path.parent
    try:
        if len(script_path.parents) >= 3:
            cwd = script_path.parents[2]
    except Exception:
        pass

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=args.full_pipeline_timeout,
        )
    except subprocess.TimeoutExpired:
        return {'ok': False, 'reason': 'full_pipeline_timeout'}
    except Exception as exc:
        return {'ok': False, 'reason': f'full_pipeline_exec_error:{exc}'}

    payload = extract_json_object(proc.stdout)
    if payload is None:
        details = proc.stderr.strip() or proc.stdout.strip()
        return {
            'ok': False,
            'reason': 'full_pipeline_invalid_json',
            'returncode': proc.returncode,
            'details': details[:400],
        }

    has_disease, label = detect_disease_from_full_pipeline(payload)
    return {
        'ok': proc.returncode == 0,
        'returncode': proc.returncode,
        'has_disease': has_disease,
        'label': label,
        'payload': payload,
    }


def read_gps_snapshot(gps_provider):
    if gps_provider is None:
        return {'lat': None, 'lon': None, 'source': 'disabled', 'fix': False}

    try:
        return gps_provider.get_position(timeout_seconds=2.0, allow_fallback=True)
    except Exception as exc:
        logging.warning('GPS read failed: %s', exc)
        return {'lat': None, 'lon': None, 'source': 'error', 'fix': False}


def usb_capture_loop(stop_event, frame_buffer, stream_client, usb_device, usb_width, usb_height, usb_fps):
    if cv2 is None:
        logging.warning('cv2 import failed, USB capture disabled.')
        return

    try:
        cap, source = open_usb_camera(usb_device, usb_width, usb_height, usb_fps)
        logging.info('USB camera opened from %s', source)
    except Exception as exc:
        logging.warning('Optional USB camera stream unavailable: %s', exc)
        return

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            frame_buffer.update(frame)

            try:
                stream_client.send_frame(frame, time.time())
            except Exception as exc:
                logging.warning('Optional USB stream send failed: %s', exc)
    finally:
        cap.release()
        logging.info('USB capture loop stopped.')


def ai_monitor_loop(stop_event, frame_buffer, gps_provider, plant_reporter, args):
    last_report_ts = 0.0

    try:
        while not stop_event.is_set():
            frame, _ts = frame_buffer.get()
            if frame is None:
                time.sleep(args.ai_interval)
                continue

            gate_triggered = False
            if args.disable_ppo_plant_gate:
                gate_triggered = True
            else:
                gate_triggered = read_ppo_plant_flag(args.ppo_plant_flag_file) == 1

            if args.force_anomaly:
                gate_triggered = True

            if not gate_triggered:
                time.sleep(args.ai_interval)
                continue

            if not args.disable_ppo_plant_gate:
                clear_ppo_plant_flag(args.ppo_plant_flag_file)

            pipeline_result = run_full_pipeline_on_frame(frame, args)
            if not pipeline_result.get('ok'):
                logging.warning('Full pipeline did not succeed: %s', pipeline_result)
                time.sleep(args.ai_interval)
                continue

            has_disease = bool(pipeline_result.get('has_disease'))
            if args.force_anomaly and not has_disease:
                has_disease = True

            if has_disease:
                now = time.monotonic()
                if now - last_report_ts >= args.anomaly_cooldown:
                    gps = read_gps_snapshot(gps_provider)
                    ai_result = {
                        'is_sick': True,
                        'is_anomaly': True,
                        'label': pipeline_result.get('label'),
                        'score': 1.0,
                        'source': 'full_pipeline_cli',
                        'full_pipeline': pipeline_result.get('payload'),
                    }
                    plant_reporter.send(frame=frame, gps_data=gps, ai_result=ai_result)
                    last_report_ts = now
                else:
                    logging.info('Disease detected but cooldown active, skipping send.')
            else:
                logging.info('Plant flag triggered but full pipeline reports no disease.')

            time.sleep(args.ai_interval)
    finally:
        if gps_provider is not None:
            gps_provider.close()
        logging.info('AI monitor loop stopped.')


def start_ppo_process(ppo_script_path):
    ppo_script = Path(ppo_script_path)
    if not ppo_script.is_absolute():
        ppo_script = (Path(__file__).resolve().parent / ppo_script).resolve()

    if not ppo_script.exists():
        raise FileNotFoundError(f'PPO script not found: {ppo_script}')

    logging.info('Starting PPO rover from %s', ppo_script)
    return subprocess.Popen([sys.executable, str(ppo_script)], cwd=str(ppo_script.parent))


def stop_ppo_process(process):
    if process is None:
        return
    if process.poll() is not None:
        return

    try:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=8)
        return
    except Exception:
        pass

    try:
        process.terminate()
        process.wait(timeout=5)
        return
    except Exception:
        pass

    process.kill()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )

    if args.test_plant_api:
        return run_plant_api_test(args)

    ppo_enabled = not args.disable_ppo
    usb_enabled = not args.disable_usb
    ai_enabled = not args.disable_ai

    if not any((ppo_enabled, usb_enabled, ai_enabled)):
        logging.error('All components are disabled. Enable at least one component or use --test-plant-api.')
        return 2

    if ai_enabled and not usb_enabled:
        logging.warning('AI loop is enabled but USB capture is disabled; no frames will be available.')

    if ai_enabled and not args.disable_ppo_plant_gate and not ppo_enabled:
        logging.info('PPO plant gate is enabled while PPO is disabled; expecting external writer to update flag file.')

    stop_event = threading.Event()
    frame_buffer = LatestFrameBuffer()

    stream_client = OptionalUSBStreamClient(
        laptop_ip=args.laptop_ip,
        port=args.laptop_port,
        endpoint=args.stream_endpoint,
        enabled=(not args.disable_usb_stream),
    )

    plant_reporter = PlantReporter(
        api_url=args.plant_api_url,
        timeout_seconds=args.plant_api_timeout,
        enabled=(not args.disable_anomaly_report),
    )

    gps_provider = None
    if ai_enabled and not args.disable_gps:
        gps_provider = GPSProvider(
            port=args.gps_port,
            baud=args.gps_baud,
            fallback_file=args.gps_fallback_file,
        )
    elif ai_enabled:
        logging.info('GPS disabled for AI anomaly reporting.')

    ppo_process = None
    if ppo_enabled:
        ensure_flag_file(args.ppo_plant_flag_file)
        clear_ppo_plant_flag(args.ppo_plant_flag_file)
        ppo_process = start_ppo_process(args.ppo_script)
    else:
        logging.info('PPO process disabled.')

    usb_thread = None
    if usb_enabled:
        usb_thread = threading.Thread(
            target=usb_capture_loop,
            args=(
                stop_event,
                frame_buffer,
                stream_client,
                args.usb_device,
                args.usb_width,
                args.usb_height,
                args.usb_fps,
            ),
            daemon=True,
            name='usb-capture',
        )
        usb_thread.start()
    else:
        logging.info('USB capture disabled.')

    ai_thread = None
    if ai_enabled:
        ai_thread = threading.Thread(
            target=ai_monitor_loop,
            args=(
                stop_event,
                frame_buffer,
                gps_provider,
                plant_reporter,
                args,
            ),
            daemon=True,
            name='ai-monitor',
        )
        ai_thread.start()
    else:
        logging.info('AI monitor disabled.')

    return_code = 0
    try:
        while True:
            if ppo_process is not None:
                rc = ppo_process.poll()
                if rc is not None:
                    return_code = rc
                    logging.info('PPO rover exited with code %s', rc)
                    break
            time.sleep(0.5)
    except KeyboardInterrupt:
        logging.info('Keyboard interrupt received, shutting down orchestrator.')
    finally:
        stop_event.set()

        if usb_thread is not None:
            usb_thread.join(timeout=2.0)
        if ai_thread is not None:
            ai_thread.join(timeout=2.0)

        stop_ppo_process(ppo_process)

    return return_code


if __name__ == '__main__':
    raise SystemExit(main())
