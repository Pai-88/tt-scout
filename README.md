# tt-scout

Table tennis match analysis from one phone on a tripod. Film a match from the side of the table, and tt-scout finds every point
and who won it, keeps score in games to 11, measures each shot's speed in 3D with its own error bar, and measures each player's
posture at every hit. The result is one report per match, with a clip of every point and the whole match as one video, plus a
profile page per player across matches. Everything runs offline on your own machine, and nothing is uploaded.

<p align="center">
  <img src="docs/media/rally.gif" width="720" alt="A rally with the ball trail, bounces, score and shot speed drawn by tt-scout">
</p>
<p align="center"><sub>One rally from an OpenTTGames league match as tt-scout draws it: ball trail coloured by speed, bounces,
net crossings, the score it keeps and each shot's measured speed. See the <a href="https://paingheinhtet.com/tt-scout/report.html">example report</a>,
with a clip of every point.</sub></p>

## How it works

1. **The table is the calibration.** A table is exactly 2.74 x 1.525 m, so its four corners are enough to recover where the
   phone stood and its focal length (OpenCV `solvePnP`, scanning the focal length). No checkerboard is needed, and every
   measurement comes out in metres. The corners are found automatically: blue and green tables by colour; any other table by
   growing its surface out from the middle of the picture until it stops at the white edge line, then snapping to that line
   (`tt_scout/tablefind.py`).

   <img src="docs/media/table_calibration.jpg" width="560" alt="Table edges found automatically and outlined in green">

2. **Ball tracking** is classical: background subtraction plus a constant-velocity tracker (`detector.py`, `tracker.py`). It
   runs on a laptop, and when it goes wrong you can find the exact frame and see why.
3. **Events and points.** Bounces, racket hits and net crossings come from kinks in the ball's path. A small state machine
   turns them into points, the server and the winner (`events.py`, `points_v1.py`, `scoring.py`).
4. **Speed from physics, not pixels.** Each shot's flight is fitted in 3D through the calibrated camera, with gravity, air drag
   and spin, anchored where the ball bounced (Levenberg-Marquardt, `flight.py`). A speed is only counted when the hit was
   actually seen, and every measured shot carries its own error bar.
5. **Bodies.** Apple's Vision framework gives 2D and 3D skeletons (`tools/pose`, `tools/pose3d`, macOS only). They are placed
   on the table with the same camera model to measure knee bend, trunk lean and shoulder turn at each hit. These are compared
   with professional players measured the same way on the OpenTTGames matches.

## Install

Needs Python 3.10+ and ffmpeg (`brew install ffmpeg`). The skeletons need macOS with Xcode's command line tools. Everything
else runs anywhere OpenCV does.

```bash
git clone https://github.com/Pai-88/tt-scout
cd tt-scout
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Use

The upload page runs on your own machine and is only reachable from it. Drop a video in the browser, check the table corners
it found, and wait for the report:

```bash
tt-scout serve
```

Each report opens with the score, a verdict on how far the automatic count can be trusted, and every point with its clip:

<img src="docs/media/report.png" width="760" alt="Match report: score, verdict, list of points and a clip player">

The same pipeline from the command line. It converts an iPhone HDR recording, finds the table, tracks, draws the skeletons and
builds the report:

```bash
python scripts/new_match.py IMG_1234.MOV --near Alex --far Sam
```

`--near` is the player at the left end of the picture. `tt-scout analyse match.mp4` is the quicker path for a plain .mp4
(no skeletons). If the table isn't found, click its corners with `tt-scout calibrate match.mp4`.

**Filming.** Stand the phone at the side of the table, level with the net, with the whole table and both players in view,
fixed for the whole match, at 60 fps. A recording made from past the end of the table is refused before tracking starts,
because from there the nearer player hides the ball at every hit (`tt_scout/filming.py`).

## How accurate it is

| What | Result | Evidence |
|:--|:--|:--|
| Winner of each point | right on all 21 point endings that could be judged by eye in two of our recordings: 16 of 16 on one (the rules were tuned on it, so a best case) and 5 of 5 checked blind on the other | frame-by-frame hand check against the video (our recordings are not included) |
| Shot speed | every measured shot carries its own error bar, typically ±4% on our recordings; on simulated shots with noise, spin and hidden racket contacts the median error is 3% | `analysis/speed_accuracy.py` |
| Speed on a real ball | not measured yet. `tt-scout droptest` compares drops from a measured height with physics (4.15 m/s from 1.00 m) | `tt_scout/droptest.py` |
| Table corners found automatically | 8 of 8 of our side-on recordings, every corner within 1.3 px of the hand-fitted one; 12 of 12 OpenTTGames clips | `analysis/check_autocal.py` on the OpenTTGames videos |
| Ball found, points, winners (professional footage) | see the figure below | 7 held-out OpenTTGames clips |

![Classical versus learned ball detector on 7 held-out OpenTTGames clips](out_ml/results_all7.png)

**Limits.**
- One camera sees the ball's movement towards or away from it least well. A speed whose racket contact was hidden is carried
  back from the bounce and shown as approximate (~).
- The rules can't tell a ball that clips the table edge from one that just misses.
- A second game in view is filtered out, but points from those stretches are less reliable.
- Filming from behind a player is refused rather than analysed badly.

## The learned ball detector (research only)

`tt_scout/ml/` holds BallNet, a U-Net with 1.56 million parameters. It looks at three frames at once and predicts a heatmap of
where the ball is. It was trained on the OpenTTGames training matches. On the seven held-out clips it finds the ball in 98% of
labelled frames, against 77% for the classical detector, and lifts point F1 from 0.81 to 0.92 with the rest of the pipeline
unchanged. It is not used by the app: its training labels are for non-commercial use only, and it has not yet been trained on
phone footage. No weights are included. To train it, download OpenTTGames, then run `pip install -e ".[ml]"` and
`python -m tt_scout.ml.train`.

<img src="docs/media/detector_comparison.gif" width="760" alt="Classical and learned ball detectors side by side">

<sub>Left: the classical detector. Right: BallNet. OpenTTGames test_4, slowed 4x.</sub>

## Reproducing the numbers

```bash
python -m unittest discover -s tests -t .         # the test suite, no footage needed
python synth/check_pipeline.py                    # synthetic matches with known ball physics
python analysis/speed_accuracy.py                 # simulated shots through a calibrated camera
python openttgames.py data/test_2                 # after downloading OpenTTGames test_2: table + truth
python eval_openttgames.py data/test_2            # score the tracker against its labels
```

## Privacy

Recordings of people are personal data. The `.gitignore` keeps footage, reports and player profiles out of git. The
upload page only answers requests from the machine it runs on.

## Licence

The code is MIT (`LICENSE`). Files derived from the OpenTTGames and Extended OpenTT Games datasets are CC BY-NC-SA 4.0 and are
listed in `DATA_LICENSE.md`, with the citations.
