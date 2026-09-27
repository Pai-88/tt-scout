"""Is a recording filmed so tt_scout can read it? Where the phone stood, measured from the table's four corners alone, and the rule
that accepts or refuses the recording before the slow tracking starts.

Where the camera stood (camera_position on each table.json, `python analysis/check_camera_ok.py`):
    own side-on recordings (8)            2 to 4 deg off the net line, 3.5 to 3.8 m from the net, 1.0 m up    works
    OpenTTGames, 12 professional clips    0 to 8 deg off, 4.0 to 4.4 m, 0.9 to 1.6 m up                          works
    IMG_3140, filmed from behind a player 63 deg off, 3.9 m, 0.67 m up, 2.1 m past the table's end             does not: the
        tracker found 5 of 6 rallies but lost the ball at every one of that player's hits, and a second tracker built for the angle did worse
"""
import math
from .config import TABLE_LENGTH as L, TABLE_WIDTH as W


def camera_position(table, width, height):
    """Where the phone stood, in metres, from the corners (tt_scout.camera: focal length scanned, then solvePnP). x runs along the
    table from its left end, y across it (negative = the camera's side), z up from the playing surface. Also off_deg (0 = level
    with the net, 90 = straight behind one end), dist_m (from the centre of the net, along the floor) and down_deg (how steeply
    the camera looks down at the net)."""
    from .camera import Camera
    cam = Camera.from_table_pnp(table, width, height)
    x, y, z = (float(v) for v in cam.C)
    dx, dy = x - L / 2, y - W / 2
    return dict(x=round(x, 2), y=round(y, 2), z=round(z, 2), off_deg=round(math.degrees(math.atan2(abs(dx), abs(dy))), 1),
                dist_m=round(math.hypot(dx, dy), 2), down_deg=round(math.degrees(math.atan2(z, math.hypot(dx, dy))), 1))


def camera_ok(pos):
    """Accept a recording only when the phone stood beside the table, between its two ends (0 <= x <= 2.74 m along it). From there
    both players are seen side-on and the ball stays in view at every hit; past an end, the near player's body hides it at theirs
    (IMG_3140 stood 2.1 m past an end). Every recording that worked stood 1.1 to 1.9 m from the left end. At the usual 3.5 to 4.5 m
    back this is about 17 to 21 deg off the net line; a fixed angle would not follow the distance, the table's own length does.
    Returns (True, "") or (False, one sentence telling the person how to film it again)."""
    if 0.0 <= pos["x"] <= L:
        return True, ""
    past = -pos["x"] if pos["x"] < 0 else pos["x"] - L
    return False, (f"The phone stood {past:.1f} m past the end of the table ({pos['off_deg']:.0f} degrees off the net line), where the "
                   f"nearer player hides the ball; film from the side of the table, level with the net.")
