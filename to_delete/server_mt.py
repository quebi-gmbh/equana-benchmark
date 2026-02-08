#!/usr/bin/env python3
"""
HTTP server with Cross-Origin headers for SharedArrayBuffer support.

Required for WebAssembly pthreads/multithreading.
"""

import http.server
import socketserver

PORT = 8080

class COOPCOEPHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Required headers for SharedArrayBuffer
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()

    def do_GET(self):
        # Set correct MIME types
        if self.path.endswith('.wasm'):
            self.send_response(200)
            self.send_header('Content-Type', 'application/wasm')
            self.end_headers()
            with open(self.path[1:], 'rb') as f:
                self.wfile.write(f.read())
            return
        super().do_GET()

with socketserver.TCPServer(("", PORT), COOPCOEPHandler) as httpd:
    print(f"Serving at http://localhost:{PORT}")
    print(f"SharedArrayBuffer enabled (COOP/COEP headers set)")
    print(f"Press Ctrl+C to stop")
    httpd.serve_forever()
