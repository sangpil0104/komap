# Model326 + Model270 validation error-map gallery

This gallery uses the already-saved best Model326+Model270 validation route. The
prediction export verifies all 20 per-image scores against the recorded route
scores before writing masks. It reads validation data only; no test data is used.

From the workspace root, serve the existing `compare_web` directory:

```bash
python3 -m http.server 8765 --directory compare_web
```

Then open <http://127.0.0.1:8765/model326/>.
