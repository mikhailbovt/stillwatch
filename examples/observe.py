"""Run against a local server; install the standalone SDK first."""

import argparse
import os

from stillwatch_sdk import SourceSelection, Stillwatch

parser = argparse.ArgumentParser()
parser.add_argument("video")
parser.add_argument("--url", default="http://127.0.0.1:8765")
args = parser.parse_args()

with Stillwatch(args.url, token=os.getenv("STILLWATCH_API_TOKEN")) as client:
    uploaded = client.upload_video(args.video)
    client.select_source(SourceSelection(kind="file", file_id=uploaded.file_id))
    try:
        for snapshot in client.updates():
            for event in snapshot.get("events", []):
                if event["analysis_status"] in ("REVIEW", "URGENT", "UNKNOWN", "UNAVAILABLE"):
                    print(event["incident_id"], event["analysis_status"], event.get("policy_reason"))
            if snapshot.get("capture_state") == "ended":
                break
    finally:
        client.stop_source()
