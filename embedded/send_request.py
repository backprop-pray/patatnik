#!/usr/bin/env python3
import argparse
import json
import mimetypes
import uuid
from pathlib import Path
from urllib import error, request


def build_multipart_body(latitude, longitude, image_path):
    boundary = f'----PythonMultipart{uuid.uuid4().hex}'
    boundary_bytes = boundary.encode('utf-8')
    crlf = b'\r\n'

    image_path = Path(image_path)
    image_bytes = image_path.read_bytes()
    image_name = image_path.name
    image_type = mimetypes.guess_type(image_name)[0] or 'application/octet-stream'

    parts = []

    def add_text(name, value):
        parts.append(b'--' + boundary_bytes + crlf)
        parts.append(f'Content-Disposition: form-data; name="{name}"'.encode('utf-8') + crlf + crlf)
        parts.append(str(value).encode('utf-8') + crlf)

    add_text('latitude', latitude)
    add_text('longitude', longitude)

    parts.append(b'--' + boundary_bytes + crlf)
    parts.append(
        f'Content-Disposition: form-data; name="image"; filename="{image_name}"'.encode('utf-8') + crlf
    )
    parts.append(f'Content-Type: {image_type}'.encode('utf-8') + crlf + crlf)
    parts.append(image_bytes)
    parts.append(crlf)
    parts.append(b'--' + boundary_bytes + b'--' + crlf)

    body = b''.join(parts)
    content_type = f'multipart/form-data; boundary={boundary}'
    return body, content_type


def main():
    parser = argparse.ArgumentParser(description='Send multipart request to /api/plants')
    parser.add_argument('--url', default='http://morzio.com/api/plants')
    parser.add_argument('--latitude', type=float, default=42.697)
    parser.add_argument('--longitude', type=float, default=23.321)
    parser.add_argument('--image', required=True, help='Path to image file')
    parser.add_argument('--timeout', type=float, default=20.0)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    image_path = Path(args.image).expanduser()
    if not image_path.exists() or not image_path.is_file():
        raise SystemExit(f'Image file not found: {image_path}')

    body, content_type = build_multipart_body(args.latitude, args.longitude, image_path)

    req = request.Request(
        args.url,
        data=body,
        headers={
            'Content-Type': content_type,
            'Accept': 'application/json',
            'User-Agent': 'python-multipart-client/1.0',
        },
        method='POST',
    )

    if args.verbose:
        print(f'> POST {args.url}')
        print(f'> Content-Type: {content_type}')
        print(f'> Content-Length: {len(body)}')

    try:
        with request.urlopen(req, timeout=args.timeout) as response:
            raw = response.read().decode('utf-8', errors='replace')
            print(f'< HTTP {response.status} {response.reason}')
            if args.verbose:
                for key, value in response.headers.items():
                    print(f'< {key}: {value}')
            try:
                parsed = json.loads(raw)
                print(json.dumps(parsed, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print(raw)
    except error.HTTPError as exc:
        raw = exc.read().decode('utf-8', errors='replace')
        print(f'< HTTP {exc.code} {exc.reason}')
        if args.verbose:
            for key, value in exc.headers.items():
                print(f'< {key}: {value}')
        print(raw)
        raise SystemExit(1)
    except error.URLError as exc:
        print(f'Connection error: {exc}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
