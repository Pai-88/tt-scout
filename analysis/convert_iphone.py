"""An iPhone recording -> the SDR H.264 file tt_scout reads, keeping the recording date.

    python analysis/convert_iphone.py ~/Downloads/IMG_3150.MOV data/own/IMG_3150.mp4

iPhones record 10-bit HEVC in HLG HDR by default; OpenCV decodes that without tone mapping, so colours come out washed out and the
colour rules for ball and table misfire. This maps BT.2020 HLG to BT.709, keeps the frame rate constant, and copies the file's
metadata (the recording date is what player profiles put matches in order by). Already-SDR files are re-encoded the same way.
"""
import argparse, json, pathlib, subprocess, sys


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def is_hdr(v):
    """v: the video stream from probe()."""
    return v.get("color_transfer") in ("arib-std-b67", "smpte2084") or v.get("color_primaries") == "bt2020"


def video_filter(hdr, max_height=None):
    """The ffmpeg -vf chain: HLG -> BT.709 when the file is HDR, then scaled down to max_height (the upload page's preview frame uses
    the same chain, so corners clicked on it are in the converted video's pixels)."""
    vf = "colorspace=all=bt709:iall=bt2020:itrc=bt2020-10:format=yuv420p" if hdr else "format=yuv420p"
    return vf + (f",scale=-2:'min(ih,{max_height})'" if max_height else "")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--crf", default="16")
    ap.add_argument("--max-height", type=int, help="scale down anything taller (4K -> 1080p); tt_scout is tuned on 1080p")
    a = ap.parse_args()
    info = probe(a.src)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = (int(x) for x in v.get("r_frame_rate", "60/1").split("/"))
    hdr = is_hdr(v)
    vf = video_filter(hdr, a.max_height)
    cmd = ["ffmpeg", "-v", "error", "-stats", "-y", "-i", a.src, "-map", "0:v:0", "-map", "0:a:0?", "-map_metadata", "0",
           "-vf", vf, "-fps_mode", "cfr", "-r", f"{num}/{den}", "-c:v", "libx264", "-preset", "veryfast", "-crf", a.crf,
           "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart+use_metadata_tags", a.dst]
    print(("HDR -> SDR: " if hdr else "SDR: ") + " ".join(cmd[-2:]), flush=True)
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
