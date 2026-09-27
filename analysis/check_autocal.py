"""Grade the zero-click calibration against the verified OpenTTGames calibrations (data/<clip>_table.json).
Usage: python -m analysis.check_autocal test_1 test_2 ..."""
import json, sys, time, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from tt_scout.autocal import auto_table
errs = []
for clip in sys.argv[1:]:
    ref = np.array(json.load(open(ROOT / f"data/{clip}_table.json"))["corners_px"]); t0 = time.time()
    corners, info = auto_table(ROOT / f"data/{clip}.mp4", frames=30)
    if corners is None:
        print(f"{clip}: FAILED ({info.get('reason')}; {info['n_frames']}/{info['n_tried']} frames)", flush=True); errs.append([np.nan] * 4); continue
    e = np.hypot(*(corners - ref).T)
    errs.append(e); print(f"{clip}: {info['colour']:5s} {info['n_frames']:2d}/{info['n_tried']} frames  corner error px {np.round(e, 1).tolist()}  max {e.max():5.1f}   ({time.time() - t0:.0f}s)", flush=True)
E = np.array(errs); ok = ~np.isnan(E[:, 0])
print(f"\n{ok.sum()}/{len(E)} clips calibrated; corner error over {ok.sum() * 4} corners: median {np.nanmedian(E):.1f} px, p90 {np.nanpercentile(E, 90):.1f} px, max {np.nanmax(E):.1f} px; clips with every corner <= 5 px: {int((np.nanmax(E, axis=1) <= 5).sum())}")
