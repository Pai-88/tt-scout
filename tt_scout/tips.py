"""Coaching tips from the tracked data. Product layer (clip overlays); not part of the evaluated method.

A recording holds 6 to 40 points: far too few to train a model on, and a bare win rate at that size is mostly noise. So a tip is
treated as a HYPOTHESIS about a player and has to earn its place on screen, in two steps.

  1  How strong is this one pattern?      "Black wins more points when the serve lands short": short serves won 4 of 5, the others
     2 of 6. Each win rate gets a Beta posterior (prior Beta(2, 2): two imaginary wins, two imaginary losses, so every rate is pulled
     toward a coin flip) and `sure` = P(rate_short > rate_rest | the points), which has an exact closed form.
  2  Is the best of MANY patterns still luck?      About fifteen candidates are tried per player and only the strongest are shown, so
     chance alone produces impressive ones: with the winners of the 21-point demo match shuffled at random, some candidate with four
     points a side reached sure >= 0.90 in 43 % of shuffles. So the winners are shuffled SHUFFLES times, the whole search is rerun each time, and
     `chance` = the share of shuffles whose strongest candidate is at least as sure as this one (a family-wise permutation test,
     the max-statistic correction of Westfall and Young). A tip has to beat that, not just its own posterior.

Rules that only DESCRIBE (where a serve usually lands, how lost points ended) do not depend on who won; their `chance` is simply
1 - P(share > 1/2). Shown as a SCOUT TIP when chance <= CHANCE_TIP and as an EARLY READ when chance <= CHANCE_EARLY; never with fewer
than MIN_N points on the thinner side (MIN_N_DESCRIBE for a share), which is what keeps "won 2 of 2" off the screen (its posterior alone would be 0.90).

The counts are tt_scout's own, and its winner call is right about three times in four on held-out footage, so a tip is a lead to
check against the clips, never a verdict. match_tips() returns nothing at all when the run's confidence verdict is "unreliable".

Nothing here says left or right: width is "wide" (the outer thirds) or "middle", which the calibration click order cannot mirror.

    tips = match_tips(points, quality)      # [Tip], the least likely to be luck first
    per_clip = assign(tips, points)         # {point id: Tip}; a clip only gets a tip that it illustrates

To add a rule: write a function (ctx) -> iterable of Tip that yields EVERY candidate with MIN_N_EARLY points a side (no bar of its
own), and append it to OUTCOME_RULES if it compares win rates, to DESCRIBE_RULES if it does not. See Ctx for what a rule is given.
"""
import math, random
from dataclasses import dataclass, field
from .config import TABLE_WIDTH as W, NET_X
from .stats import DEPTH, bucket, oriented, player_end

PRIOR = (2, 2)                          # Beta prior on every rate: two imaginary wins, two imaginary losses
CHANCE_TIP, CHANCE_EARLY = 0.10, 0.35   # how often luck alone does as well: at most 1 match in 10 for a tip, about 1 in 3 for an early read
SHUFFLES = 400                          # winner shuffles behind `chance`
MIN_N_TIP, MIN_N_EARLY = 4, 3           # points needed on the thinner side of a comparison
MIN_N_DESCRIBE = 5                      # points needed before a share ("served short 4 of 5") is worth saying
LONG_RALLY = 5                          # shots over the net from which a rally counts as long
MAX_TIPS = 8
MAX_SHOWS = 3                           # clips that carry the same note, spread over the match
DEPTH_WORD = {"short": "short", "mid": "half-long", "deep": "deep"}


def _lbeta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def prob_greater(w1, n1, w2, n2, prior=PRIOR):
    """P(rate1 > rate2 | w1 of n1 and w2 of n2), independent Beta posteriors. Exact for whole-number counts:
    sum_{i < a1} B(a2 + i, b1 + b2) / ((b1 + i) B(1 + i, b1) B(a2, b2))."""
    a1, b1 = prior[0] + w1, prior[1] + n1 - w1
    a2, b2 = prior[0] + w2, prior[1] + n2 - w2
    return float(sum(math.exp(_lbeta(a2 + i, b1 + b2) - math.log(b1 + i) - _lbeta(1 + i, b1) - _lbeta(a2, b2)) for i in range(int(a1))))


def prob_majority(k, n, prior=PRIOR):
    """P(share > 1/2 | k of n), Beta posterior. For whole numbers I_x(a, b) = P(Binomial(a + b - 1, x) >= a), so this is exact too."""
    a, b = prior[0] + k, prior[1] + n - k
    m = a + b - 1
    return float(1.0 - sum(math.comb(m, j) for j in range(a, m + 1)) / 2 ** m)


@dataclass
class Tip:
    rule: str
    player: str                          # who it talks to
    head: str                            # the suggestion, a few words
    evidence: str                        # the counts behind it
    sure: float                          # posterior probability of this one pattern, on its own
    n: int                               # points on the thinner side of the comparison
    shows_on: list = field(default_factory=list)     # ids of the points that illustrate it
    chance: float = 1.0                  # how often luck alone does as well, over the whole search (set by match_tips)
    family: str = "describe"             # "outcome" = compares win rates, judged against shuffled winners; "describe" = a share

    @property
    def level(self):
        return "tip" if self.chance <= CHANCE_TIP and self.n >= MIN_N_TIP else "early"

    @property
    def key(self):
        return f"{self.rule}:{self.player}:{self.head}"

    def to_json(self):
        return dict(rule=self.rule, player=self.player, head=self.head, evidence=self.evidence, sure=round(self.sure, 3),
                    chance=round(self.chance, 3), family=self.family, n=self.n, level=self.level, shows_on=self.shows_on)


class Ctx:
    """What a rule gets: the points, the two names, and the points whose winner is known (win rates use only those)."""

    def __init__(self, points):
        self.points = points
        self.names = []
        for pt in points:
            for k in ("near_player", "far_player"):
                if pt.get(k) and pt[k] not in self.names:
                    self.names.append(pt[k])
        self.names = self.names[:2]
        self.known = [pt for pt in points if pt.get("winner_name") in self.names]

    def other(self, name):
        return next((n for n in self.names if n != name), None)

    def last_hitter(self, pt):
        """Whose shot crossed the net last: the server's if the count is odd, the receiver's if it is even."""
        n, srv = pt.get("n_crossings", 0), pt.get("server")
        if not n or srv not in self.names:
            return None
        return srv if n % 2 == 1 else self.other(srv)


def serve_landing(pt):
    """(depth bucket, "wide" | "middle") of the serve on the receiver's half, or None when that landing was not seen."""
    srv = pt.get("server")
    l = next((l for l in pt.get("landings", []) if l.get("shot") == 1), None)
    if l is None or srv is None:
        return None
    x, y = oriented(l["x_m"], l["y_m"], player_end(pt, srv))
    if x <= NET_X:
        return None
    return bucket(x - NET_X, DEPTH), ("wide" if abs(y - W / 2) > W / 6 else "middle")


def _wins(pts, name):
    return sum(1 for pt in pts if pt.get("winner_name") == name)


def _ids(pts):
    return [pt["id"] for pt in pts]


def _compare(ins, outs, name):
    """(sure that `name` does better in `ins`, sure that they do worse, n on the thinner side), or None below MIN_N_EARLY."""
    n = min(len(ins), len(outs))
    if n < MIN_N_EARLY:
        return None
    up = prob_greater(_wins(ins, name), len(ins), _wins(outs, name), len(outs))
    return up, 1.0 - up, n


def rule_serve(ctx):
    """To the server: which serve depth or width is paying, or is not. At most one tip per player and dimension."""
    nouns = {"short": "short serves", "mid": "half-long serves", "deep": "deep serves", "wide": "wide serves", "middle": "serves into the middle"}
    adverb = dict(DEPTH_WORD, wide="wide", middle="into the middle")
    for P in ctx.names:
        rows = [(serve_landing(pt), pt) for pt in ctx.known if pt.get("server") == P]
        rows = [(sl, pt) for sl, pt in rows if sl]
        for dim, buckets in ((0, ("short", "mid", "deep")), (1, ("wide", "middle"))):
            best = None
            for b in buckets:
                ins = [pt for sl, pt in rows if sl[dim] == b]; outs = [pt for sl, pt in rows if sl[dim] != b]
                c = _compare(ins, outs, P)
                if not c:
                    continue
                ev = f"{nouns[b].capitalize()} won {_wins(ins, P)} of {len(ins)} · the others {_wins(outs, P)} of {len(outs)}"
                for sure, good in ((c[0], True), (c[1], False)):          # on equal evidence "keep doing" is heard before "do less"
                    if best is None or sure > best.sure + 1e-9:
                        won_it = [pt for pt in ins if (pt["winner_name"] == P) == good]
                        best = Tip("serve_depth" if dim == 0 else "serve_width", P, f"Keep serving {adverb[b]}" if good else f"Serve {adverb[b]} less often",
                                   ev, sure, c[2], _ids(won_it or ins))
            if best:
                yield best


def rule_rally(ctx):
    """Does this player do better once the rally gets long, or when it stays short?"""
    for P in ctx.names:
        long_ = [pt for pt in ctx.known if pt.get("n_crossings", 0) >= LONG_RALLY]
        short = [pt for pt in ctx.known if pt.get("n_crossings", 0) < LONG_RALLY]
        c = _compare(long_, short, P)
        if not c:
            continue
        counts = (_wins(long_, P), len(long_), _wins(short, P), len(short))
        if c[0] >= c[1]:
            yield Tip("rally", P, "Make the rally long", "{4}+ shots: won {0} of {1} · shorter: won {2} of {3}".format(*counts, LONG_RALLY), c[0], c[2],
                      _ids([pt for pt in long_ if pt["winner_name"] == P] or long_))
        else:
            yield Tip("rally", P, "Finish the point early", "Under {4} shots: won {2} of {3} · longer: won {0} of {1}".format(*counts, LONG_RALLY), c[1], c[2],
                      _ids([pt for pt in short if pt["winner_name"] == P] or short))


def rule_lost_points(ctx):
    """How this player's lost points end: own shot long or wide, or a ball that did not come back, and where that ball landed."""
    for P in ctx.names:
        O = ctx.other(P)
        lost = [pt for pt in ctx.known if pt["winner_name"] != P]
        if len(lost) < MIN_N_DESCRIBE:
            continue
        long_ = [pt for pt in lost if pt.get("ending") == "long" and ctx.last_hitter(pt) == P]
        sure = prob_majority(len(long_), len(lost))
        if len(long_) * 2 > len(lost):
            yield Tip("lost_long", P, "Aim further inside the lines", f"{len(long_)} of {len(lost)} lost points: own shot went long or wide", sure, len(lost), _ids(long_))
        beaten = []                                                     # (depth bucket of the ball that was not returned, point)
        for pt in lost:
            if pt.get("ending") in ("not_returned", "double_bounce") and ctx.last_hitter(pt) == O and pt.get("landings"):
                l = pt["landings"][-1]
                x, _ = oriented(l["x_m"], l["y_m"], player_end(pt, O))
                if x > NET_X:
                    beaten.append((bucket(x - NET_X, DEPTH), pt))
        if len(beaten) < MIN_N_DESCRIBE:
            continue
        for b, head in (("deep", "Give yourself room for the deep ball"), ("short", "Step in to the short ball")):
            ins = [pt for d, pt in beaten if d == b]
            sure = prob_majority(len(ins), len(beaten))
            if len(ins) * 2 > len(beaten):
                yield Tip("beaten_" + b, P, head, f"{len(ins)} of {len(beaten)} balls not returned landed {b}", sure, len(beaten), _ids(ins))


def rule_read_serve(ctx):
    """To the receiver: where the opponent's serve usually lands."""
    for P in ctx.names:
        O = ctx.other(P)
        rows = [(serve_landing(pt), pt) for pt in ctx.points if pt.get("server") == O]
        rows = [(sl, pt) for sl, pt in rows if sl]
        if len(rows) < MIN_N_DESCRIBE:
            continue
        for b, word in DEPTH_WORD.items():
            ins = [pt for sl, pt in rows if sl[0] == b]
            sure = prob_majority(len(ins), len(rows))
            if len(ins) * 2 > len(rows):
                yield Tip("read_serve", P, f"Expect the {word} serve", f"{O} served {word} {len(ins)} of {len(rows)} times", sure, len(rows), _ids(ins))


def rule_serve_return(ctx):
    """Is the gap on this player's own serve or on the return?"""
    for P in ctx.names:
        sv = [pt for pt in ctx.known if pt.get("server") == P]; rc = [pt for pt in ctx.known if pt.get("server") == ctx.other(P)]
        c = _compare(sv, rc, P)
        if not c:
            continue
        ev = f"Won {_wins(sv, P)} of {len(sv)} on serve · {_wins(rc, P)} of {len(rc)} on return"
        if c[0] >= c[1]:
            yield Tip("serve_return", P, "Work on the return of serve", ev, c[0], c[2], _ids([pt for pt in rc if pt["winner_name"] != P] or rc))
        else:
            yield Tip("serve_return", P, "Get more from the serve", ev, c[1], c[2], _ids([pt for pt in sv if pt["winner_name"] != P] or sv))


OUTCOME_RULES = [rule_serve, rule_rally, rule_serve_return]        # compare win rates: judged against shuffled winners
DESCRIBE_RULES = [rule_lost_points, rule_read_serve]               # say where or how: judged on their own posterior


def _strongest_by_luck(ctx, shuffles, seed):
    """Sorted `sure` of the strongest outcome candidate in each of `shuffles` copies of the match with the winners dealt out at random."""
    rng = random.Random(seed)
    winners = [pt["winner_name"] for pt in ctx.known]
    out = []
    for _ in range(shuffles):
        rng.shuffle(winners)
        fake = Ctx([dict(pt, winner_name=w) for pt, w in zip(ctx.known, winners)])
        out.append(max((t.sure for rule in OUTCOME_RULES for t in rule(fake)), default=0.0))
    return sorted(out)


def match_tips(points, quality=None, shuffles=SHUFFLES, seed=0):
    """[Tip], the least likely to be luck first. Empty when the run is unreliable or the two players are not named."""
    if quality and quality.get("level") == "unreliable":
        return []
    ctx = Ctx(points)
    if len(ctx.names) < 2:
        return []
    outcome = [t for rule in OUTCOME_RULES for t in rule(ctx)]
    if outcome:
        null = _strongest_by_luck(ctx, shuffles, seed)
        for t in outcome:
            t.family = "outcome"
            t.chance = sum(1 for m in null if m >= t.sure - 1e-12) / len(null)
    describe = [t for rule in DESCRIBE_RULES for t in rule(ctx)]
    for t in describe:
        t.chance = 1.0 - t.sure
    tips = [t for t in outcome + describe if t.chance <= CHANCE_EARLY]
    tips.sort(key=lambda t: (t.chance, -t.sure, t.rule, t.player))
    return tips[:MAX_TIPS]


def assign(tips, points):
    """{point id: Tip}. A clip only gets a tip that its point illustrates, and one tip rides on at most MAX_SHOWS clips, spread over the
    points that show it (first, middle, last), so stepping through a match does not repeat the same note on every clip. Where two
    tips want one clip the surer keeps it and the other moves to its nearest free point."""
    order = {pt["id"]: i for i, pt in enumerate(points)}
    out = {}
    for t in sorted(tips, key=lambda t: (t.chance, -t.sure)):
        ids = sorted((i for i in t.shows_on if i in order), key=order.get)
        k = min(MAX_SHOWS, len(ids))
        for j in range(k):
            want = ids[int(round(j * (len(ids) - 1) / max(1, k - 1)))]
            free = [i for i in ids if i not in out]
            if not free:
                break
            out[min(free, key=lambda i: abs(order[i] - order[want]))] = t
    return out


if __name__ == "__main__":
    import json, pathlib, sys
    d = pathlib.Path(sys.argv[1])
    pts = json.loads((d / "rallies.json").read_text())
    q = json.loads((d / "quality.json").read_text()) if (d / "quality.json").exists() else None
    tips = match_tips(pts, q)
    per = assign(tips, pts)
    for t in tips:
        print(f"{t.level:5s} chance {t.chance:.2f} sure {t.sure:.2f} n={t.n}  {t.player}: {t.head}  [{t.evidence}]  on {t.shows_on}")
    print("on clips:", {pid: t.head for pid, t in sorted(per.items())})
    print(f"{len(tips)} tips; {len(per)} of {len(pts)} clips carry one" + ("" if tips or not q else f"  (verdict: {q.get('level')})"))
