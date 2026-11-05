#!/usr/bin/env python3
"""Static viewer server that also accepts POST /save: writes the A/B answers JSON to
./answers/ab-<seed>.json (atomic). Usage: python3 server.py [port]  (binds 127.0.0.1)."""
import json, os, sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class H(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path.split("?")[0] != "/save":
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            d = json.loads(body)
            os.makedirs("answers", exist_ok=True)
            path = f"answers/ab-{d.get('seed', 'x')}.json"
            with open(path + ".tmp", "wb") as f:
                f.write(body)
            os.replace(path + ".tmp", path)
        except Exception as e:
            self.send_error(400, str(e))
            return
        msg = f"{os.path.abspath(path)} ({len(d.get('answers', []))} answers)".encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(msg)))
        self.end_headers()
        self.wfile.write(msg)


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 8765), H).serve_forever()
