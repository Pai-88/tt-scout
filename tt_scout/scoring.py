"""Point-winner logic from the event sequence.

Uses BOUNCES and NET CROSSINGS only. Racket hits are over-detected on real footage, so they are ignored;
crossings were found 39/39 times and bounces ~90 % on the two OpenTTGames clips, so the rules lean on them.

The idea: every crossing puts the ball on one side of the table. From then until the next crossing the ball
is that player's problem. How the LAST such stretch ends decides the point:

    bounces on that side after the last crossing   meaning                                  winner
    ------------------------------------------------------------------------------------------------
    1 or more                                      the ball landed on their side and never    the other player
                                                   came back: missed return, into the net,
                                                   double bounce, ball died
    0                                              the ball crossed to them but never landed: this player
                                                   the other player's shot went long or wide

    no crossing at all in the rally                the serve never got over the net           the receiver

Bounces after the last crossing are counted whatever side the detector put them on: without another
crossing the ball cannot have changed sides, and bounces near the net are the ones most often assigned
to the wrong side.

Returns "near" | "far" | None.  None when the rally has no bounces or the ball was seen in too few frames.

Known blind spots (write these down for the coach):
  * a missed bounce in the final stretch flips "0 bounces" to the wrong player;
  * a let (serve clips the net and goes over) is scored as a normal point;
  * a volley is scored as if the ball had landed (the volleying player should lose);
  * if rally segmentation glues two points together, this returns the winner of the LAST one.
"""
# ------------------------------------------------------------------------------------------------------------
# REWRITE TARGET (research track, item 6). who_won below and the two "late bounce" fixes in
# events.py::segment_points (2026-09-15) are to be re-derived and re-implemented from the rules above;
# tests/test_scoring.py pins the
# behaviour (python -m unittest discover -s tests -v). git tag v0-scouting keeps this version as the frozen
# baseline for the ablation tables.
# ------------------------------------------------------------------------------------------------------------


def other(side):
    return "far" if side == "near" else "near"


def who_won(rally, min_coverage=0.3):
    if rally.get("coverage", 1.0) < min_coverage:
        return None
    evs = [e for e in rally.get("events", []) if e["kind"] in ("bounce", "net")]
    bounces = [e for e in evs if e["kind"] == "bounce"]
    if not bounces:
        return None
    crossings = [e for e in evs if e["kind"] == "net"]
    if not crossings:
        server = rally.get("serve_side") or bounces[0]["side"]
        return other(server)                                   # serve never crossed the net
    last = crossings[-1]
    ball_side = last["side"]                                   # side the ball crossed TO
    landed = sum(1 for e in bounces if e["t"] > last["t"])
    return other(ball_side) if landed >= 1 else ball_side
