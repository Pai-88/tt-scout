"""Find the table in an uploaded video for the upload page: frames taken through the same colour conversion and scaling as the video
the pipeline reads (so the corners are in its pixels), then tt_scout.tablefind. Writes JSON {"corners": [[x, y]] * 4 or null, ...}.
    python scripts/find_table.py upload.mov auto_table.json"""
import json, pathlib, subprocess, sys, tempfile
import numpy as np, cv2
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
import convert_iphone as conv
from tt_scout.tablefind import find_table_in_frames


def main():
    src, out = sys.argv[1], pathlib.Path(sys.argv[2])
    info = conv.probe(src); v = next(s for s in info["streams"] if s["codec_type"] == "video")
    dur = float(info.get("format", {}).get("duration") or v.get("duration") or 0)
    vf = conv.video_filter(conv.is_hdr(v), 1080)
    frames = []
    with tempfile.TemporaryDirectory() as td:
        for k, t in enumerate(np.linspace(0.03 * dur, 0.97 * dur, 25) if dur else [0.0]):
            f = pathlib.Path(td) / f"{k:02d}.png"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", src, "-frames:v", "1", "-vf", vf, str(f)], timeout=120)
            im = cv2.imread(str(f))
            if im is not None:
                frames.append(im)
    frames = [f for f in frames if f.shape == frames[0].shape] if frames else []
    c, found = find_table_in_frames(frames)
    res = dict(corners=None if c is None else [[round(float(x), 1), round(float(y), 1)] for x, y in c],
               method=found.get("method"), reason=found.get("reason"))
    out.write_text(json.dumps(res)); print(json.dumps(res))


if __name__ == "__main__":
    main()
