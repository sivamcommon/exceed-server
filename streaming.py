"""
Streaming layer entrypoint.

This file marks a dedicated place for:
- stream mode switching
- MJPEG frame generation
- source switching (camera/video)
- stream worker controls

Current implementation remains in `server.py` and is being migrated here in
small safe steps to avoid runtime regressions.
"""
