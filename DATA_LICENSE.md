# Data licence

The code in this repository is MIT licensed (see `LICENSE`). The files below are not code: they are derived from two public
datasets that are licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
(CC BY-NC-SA 4.0, https://creativecommons.org/licenses/by-nc-sa/4.0/), and they carry the same licence. You may use them for
non-commercial work, with attribution, and share what you build from them under the same terms.

| Files | Derived from |
|:--|:--|
| `data/test_*_table.json`, `data/game_*_table.json`, `data/pass_corners.json` | table corners measured on OpenTTGames videos |
| `data/test_*_truth.json`, `data/game_*_truth.json` | OpenTTGames ball positions and events, converted |
| `labels/*.points.csv` | Extended OpenTT Games point and rally-ending labels, converted |
| `tt_scout/pro_shots.json`, `tt_scout/pro_bodies.json`, `tt_scout/benchmarks.json` | shots, 3D bodies and benchmarks measured on the OpenTTGames test matches |
| `results/*.json`, `results/*.txt`, `out_ml/results_all7.png` | evaluation results on OpenTTGames |
| `docs/media/*` except `docs/media/own/` | frames and clips from the OpenTTGames test_3 and test_4 videos, with tt-scout's drawings on them |

The full videos are not included; download them from the dataset pages.

`docs/media/own/` is not covered by either licence: those pictures and clips come from our own recordings, show real people who
agreed to be shown in this repository, and are not licensed for reuse.

## Attribution

- **OpenTTGames** (OSAI): https://lab.osai.ai/datasets/openttgames. Voeikov, R., Falaleev, N., Baikulov, R. "TTNet: Real-time
  temporal and spatial video analysis of table tennis." CVPR Workshops, 2020.
- **Extended OpenTT Games**: https://github.com/moamal01/table_tennis_data (arXiv:2512.19327).

Model weights trained on these labels are research-only for the same reason, and none are included here.
