"""Knobs. Defaults were tuned on the synthetic clips in synth/; expect to retune on real footage."""
from dataclasses import dataclass
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
TABLE_LENGTH = 2.74     # m, x axis: 0 = "near" end line (the first two clicked corners), 2.74 = far end
TABLE_WIDTH = 1.525     # m, y axis: 0 = the long edge between clicked corners 1 and 4
NET_X = TABLE_LENGTH / 2
BALL_RADIUS = 0.02      # m


@dataclass
class Config:
    near_name: str = "near"         # player at the x=0 end (corners 1-2 of the calibration); shown in the report
    far_name: str = "far"
    ball_colour: str = "any"        # "orange" | "white" | "any"  (hard filter on candidate blobs)
    colour_min_frac: float = 0.3    # fraction of blob pixels that must match the colour prior
    mog_history_s: float = 4.0      # background model memory, seconds (scaled by fps)
    mog_var_threshold: float = 30.0
    area_min_frac: float = 0.3      # blob area / ball area at the far edge, lower bound
    area_max_frac: float = 4.0      # blob area / ball area at the near edge, upper bound (streaks are bigger)
    min_fill: float = 0.3           # blob area / bounding-box area
    max_aspect: float = 8.0         # a motion-streaked ball is elongated; players' edges are far longer
    gate_base_px: float = 25.0      # tracker gate = base + gate_vel_mult * |velocity|
    gate_vel_mult: float = 2.5      # 2.5 lets the ball reverse direction at a hit and stay in the gate
    init_max_jump_px: float = 200.0
    max_missed_s: float = 0.12      # seconds a track may coast before it is dropped (scaled by fps)
    init_resid_px: float = 15.0     # 3rd point must be this close to the straight-line prediction to start a track
    min_speed_px_s: float = 360.0   # a track slower than this for slow_track_s is not the ball (~1 m/s at 350 px/m)
    slow_track_s: float = 0.07
    preempt_missed: int = 2         # a coasting track loses to a fresh straight-line track after this many misses
    preempt_missed_healthy: int = 7 # ... unless it has followed a fast ball for healthy_min_frames: then it keeps waiting this long. A white
    healthy_min_frames: int = 6     # ball crossing in front of a light table further back vanishes for ~4 frames; after 2 misses the
                                    # tracker used to jump to any three dots in a line (a shoe) and lose the crossing (our 2026-09-26 videos)
    healthy_net_px: float = 150.0   # ... and only while it is expected within this many px of the net line: that is where it vanishes (in
                                    # front of the table behind); waiting anywhere else (at a racket) cost a hand-checked ending on IMG_3143
    healthy_area_frac: float = 0.45 # ... and only a ball the size of this table's ball (median blob area over its last frames, as a fraction of
                                    # the detector's ball area at this table): a ball on a table further back is 20 to 30 px against 70 to 160,
                                    # and waiting for that one cost 20 of 87 points on our 2026-09-25 recording (a game ran on the table behind)
    coast_speed_lo: float = 0.4     # while coasting, a candidate's implied speed must be within these multiples
    coast_speed_hi: float = 1.6     # of the last speed (a bounce keeps ~90 %; junk far away implies 3x)
    diff_thresh: int = 18           # grey-level change between consecutive frames that counts as "moving"
    min_moving_frac: float = 0.12   # a candidate blob must have this fraction of pixels moving (kills uncovered background)
    player_margin_px: int = 15      # reject candidates this close to a player-sized foreground blob
    vel_smooth: float = 0.8         # weight on the newest velocity estimate
    event_min_gap_s: float = 0.025  # seconds between kinks (a net clip and the bounce after it are ~0.04 s apart)
    kink_window_s: float = 0.025    # velocity before/after a frame is averaged over this long (kills centroid jitter)
    event_rel_thresh: float = 3.0   # kink must exceed this x the median kink of the segment ...
    event_cap_m_s: float = 1.5      # ... but never more than this velocity change in m/s (a real bounce is bigger)
    event_abs_thresh_px: float = 2.0  # px per frame of velocity change, minimum
    hit_min_m_s: float = 3.0        # an off-table kink must change velocity by at least this to count as a racket hit
    bounce_max_m_s: float = 12.0    # a table bounce cannot change velocity by more than this; bigger = racket (or a tracker jump)
    kink_max_m_s: float = 50.0      # nothing in the game changes velocity by more than this: tracker glitch, ignore
    max_gap_s: float = 0.25         # ball unseen for longer than this splits the track for event detection
    bounce_min_vy: float = 0.5      # px/frame: vertical velocity must be at least this much down before and up after
    table_margin_x: float = 0.06    # m, tolerance when testing "bounce inside table" (length direction)
    table_margin_y: float = 0.15    # m, width direction is the poorly resolved one from a side camera
    net_margin: float = 0.0        # m: a 'bounce' this close to the net line is a net hit (0 = off; short-game bounces land 3 cm from the net)
    rally_gap_s: float = 1.0        # no ball for this long ends the rally (fallback when no crossings are detected)
    rally_min_s: float = 0.4
    cross_max_px: float = 60.0      # both frames of a net crossing must be this close to the net line (kills tracker jumps)
    cross_debounce_s: float = 0.1   # opposite crossings closer than this are jitter at the line, not two crossings
    cross_front_s: float = 0.0      # >0: a crossing counts only if, within this many seconds of it, the ball is seen down at this table's
                                    # surface (below its far edge line) or out past its ends, where this table's players hit it. A game on
                                    # a table further back crosses this net's line in the picture but stays above and between those.
                                    # 0 = off (OpenTTGames and our 2026-09-25 short recording have one game in view).
    cross_front_margin_px: float = 40.0  # "past its ends" = further out than the far corners by this many px
    cross_front_tol_px: float = 15.0  # "down at this table" = lower in the picture than 15 px above its far edge line: a serve bouncing near
                                    # the far side line sits only ~5 px below it (our short recording lost a real point at tol 0); the
                                    # back table's ball never came lower than ~50 px above it
    cross_front_min: int = 2        # frames of such evidence needed (our 2026-09-25 recording: 2 frames and 40 px remove 173 back-table
                                    # crossings in the ten minutes that game ran and none of the 267 in the eight minutes after)
    cross_min_seen: int = 5         # fewer tracked frames than this around a crossing = too little evidence: keep it
    net_touch_s: float = 0.4        # a crossing is not real (a net touch, or jitter on the net line) when, in this long before or after it,
    net_touch_min_m: float = 0.2    # the ball never gets further than this from the net line on that side (a real shot is metres away)
    net_touch_min_seen: int = 4     # ... judged only when the ball is seen in at least this many frames of that window
    net_touch_bounce_s: float = 0.8 # also not real when the next landing within this long is on the side the ball came from ...
    net_touch_bounce_m: float = 0.35  # ... and this close to the net (the tracker often loses the ball the moment it touches the net)
    contact_shift: bool = True      # judge "is this bounce on the table" at the contact point, one ball radius below the ball's centre
                                    # (our 2026-09-25 recording: a landing on the far edge line read 7 px = 0.155 m outside from the centre)
    serve_lead_s: float = 1.0       # the serve's own-side bounce is at most this long before the first crossing
    return_timeout_s: float = 1.5   # ball landed on a side and no crossing within this = never returned, point over
    long_timeout_s: float = 1.5     # crossed and neither landed nor crossed back within this = went long, point over
    dead_time_s: float = 2.5        # after a point ends, crossings this long are ball retrieval, not play
