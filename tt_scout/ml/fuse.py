"""The learned detector beside the classical one: both look at every frame, the tracker gets one list of candidates.

The classical detector (tt_scout.detector) finds small moving blobs at full resolution. The learned one (BallNet) finds the ball
by what it looks like over three frames, at a third of the resolution. They miss in different frames: the blob detector loses a
white ball in front of something white and anything inside a player's outline, the network only knows what it was trained on.
No PyTorch is imported here: the peaks come in as plain numbers (tt_scout.ml.infer.candidates makes them).
"""
from ..detector import Candidate

MERGE_PX = 8.0          # a peak and a blob this close are one ball (a peak is placed to ~1.5 px at 1080p, a streak's centroid to ~3)


def as_candidates(learned, area_ref):
    """The network's peaks [(x, y, score)] as tracker candidates. A peak has no blob, so it is given the area a ball has at this
    table: the tracker waits for a lost ball only when what it followed was a ball of this table's size."""
    return [Candidate(float(x), float(y), float(area_ref), float(s)) for x, y, s in learned]


def fuse(classical, learned, area_ref, merge_px=MERGE_PX):
    """One frame's candidates from both detectors: every blob, and every peak that no blob explains. Where a peak and a blob are
    the same ball the blob's position is kept (it is measured at full resolution) with the better of the two scores."""
    out = [Candidate(c.x, c.y, c.area, c.score) for c in classical]
    for x, y, s in learned:
        near = min(out[:len(classical)], key=lambda c: (c.x - x) ** 2 + (c.y - y) ** 2, default=None)
        if near is not None and (near.x - x) ** 2 + (near.y - y) ** 2 <= merge_px ** 2:
            near.score = max(near.score, float(s))
        else:
            out.append(Candidate(float(x), float(y), float(area_ref), float(s)))
    return out
