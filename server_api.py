"""
HTTP API surface for the vision server.

This file is intentionally small for now and points to `server.py` where the
existing endpoints currently live. It is the migration target for route-by-route
extraction to keep a flat, file-based structure (no deep folders).
"""

# Next step migration pattern:
# 1) Move one endpoint group from server.py into this file.
# 2) Keep request/response payloads unchanged.
# 3) Import and register from server.py.
