"""
Tiny static file server for the FinReg demo frontend.
Run:  python frontend/serve.py
Then open:  http://localhost:3000
"""
import http.server, os, webbrowser

PORT = 3000
os.chdir(os.path.dirname(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_): pass   # silent

print(f"  FinReg demo → http://localhost:{PORT}")
print("  Press Ctrl-C to stop.\n")
webbrowser.open(f"http://localhost:{PORT}")
http.server.HTTPServer(("", PORT), Handler).serve_forever()
