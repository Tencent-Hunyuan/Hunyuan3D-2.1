#!/usr/bin/env python3
"""Integration test for cancellation against a running Hunyuan3D API server.

Usage:
    # 1. Successful completion (baseline — measures stage timings)
    python tests/test_integration.py --image assets/demo.png

    # 2. Cancel during surface extraction (cancel after diffusion + volume decoding)
    python tests/test_integration.py --image assets/demo.png --cancel-after 90

    # 3. Cancel during volume decoding
    python tests/test_integration.py --image assets/demo.png --cancel-after 50

    # Custom server address
    python tests/test_integration.py --image assets/demo.png --server http://10.0.0.5:8080

The script POSTs an image in a background thread.  If --cancel-after is
given, it waits that many seconds then POSTs /cancel and measures how
quickly the 409 comes back.  A cancellable surface extraction should
respond within ~1-2 s; if it takes 50+ s the subprocess fix is not active.
"""

import argparse
import os
import sys
import threading
import time

import requests


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"{ts} - integration-test - {msg}", flush=True)


def upload_image(server: str, image_path: str) -> requests.Response:
    url = f"{server}/convert-image-to-3d"
    filename = os.path.basename(image_path)
    with open(image_path, "rb") as f:
        return requests.post(url, files={"file": (filename, f, "image/png")})


def cancel(server: str) -> requests.Response:
    return requests.post(f"{server}/cancel")


def run_test(server: str, image_path: str, cancel_after: float | None) -> bool:
    """Run a single test. Returns True on pass."""

    log(f"POSTing {os.path.basename(image_path)} to {server}/convert-image-to-3d")

    convert_result: list[requests.Response | None] = [None]
    convert_error: list[Exception | None] = [None]

    def do_convert():
        try:
            convert_result[0] = upload_image(server, image_path)
        except Exception as exc:
            convert_error[0] = exc

    post_time = time.monotonic()
    t = threading.Thread(target=do_convert)
    t.start()

    if cancel_after is not None:
        log(f"Waiting {cancel_after:.0f}s before sending cancel...")
        time.sleep(cancel_after)

        log("POSTing /cancel")
        cancel_time = time.monotonic()
        cancel_resp = cancel(server)
        log(f"Cancel response: {cancel_resp.status_code} {cancel_resp.json()}")

        t.join(timeout=120)
        end_time = time.monotonic()

        if convert_error[0]:
            log(f"FAIL: convert raised {convert_error[0]}")
            return False

        resp = convert_result[0]
        if resp is None:
            log("FAIL: convert thread did not complete within 120s")
            return False

        cancel_to_response = end_time - cancel_time
        total_time = end_time - post_time
        log(f"Convert response: {resp.status_code}")
        log(f"Total time:           {total_time:.2f}s")
        log(f"Cancel-to-response:   {cancel_to_response:.2f}s")

        if resp.status_code == 409:
            if cancel_to_response < 5.0:
                log("PASS: 409 received promptly after cancel")
                return True
            else:
                log(
                    f"FAIL: 409 received but took {cancel_to_response:.2f}s "
                    f"(expected <5s — subprocess termination may not be working)"
                )
                return False
        elif resp.status_code == 200:
            log(
                f"FAIL: got 200 instead of 409 — cancel was sent too late "
                f"(processing completed before cancel took effect). "
                f"Try a smaller --cancel-after value."
            )
            return False
        else:
            log(f"FAIL: unexpected status {resp.status_code}")
            try:
                log(f"  Body: {resp.text[:500]}")
            except Exception:
                pass
            return False
    else:
        # No cancel — just let it complete and report timing
        log("No --cancel-after specified, waiting for completion...")
        t.join(timeout=600)
        end_time = time.monotonic()

        if convert_error[0]:
            log(f"FAIL: convert raised {convert_error[0]}")
            return False

        resp = convert_result[0]
        if resp is None:
            log("FAIL: convert thread did not complete within 600s")
            return False

        total_time = end_time - post_time
        log(f"Convert response: {resp.status_code}")
        log(f"Total time: {total_time:.2f}s")

        if resp.status_code == 200:
            content_length = len(resp.content)
            log(f"PASS: received GLB ({content_length:,} bytes) in {total_time:.2f}s")

            # Save the GLB for inspection
            out_path = f"test_output_{int(time.time())}.glb"
            with open(out_path, "wb") as f:
                f.write(resp.content)
            log(f"Saved to {out_path}")
            return True
        else:
            log(f"FAIL: expected 200, got {resp.status_code}")
            try:
                log(f"  Body: {resp.text[:500]}")
            except Exception:
                pass
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Integration test for Hunyuan3D API cancellation"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to PNG image to upload",
    )
    parser.add_argument(
        "--server",
        default="http://localhost:8080",
        help="API server URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--cancel-after",
        type=float,
        default=None,
        help="Seconds after POST to send /cancel. "
        "Omit to let the request complete normally.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Check server is reachable
    try:
        r = requests.get(f"{args.server}/health", timeout=5)
        log(f"Server health: {r.json()}")
    except Exception as exc:
        print(f"Error: cannot reach server at {args.server}: {exc}", file=sys.stderr)
        sys.exit(1)

    passed = run_test(args.server, args.image, args.cancel_after)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
