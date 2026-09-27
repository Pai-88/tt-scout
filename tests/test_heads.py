"""Heads found against an empty-hall plate, the landing map's size choice, and balloons that keep off every head."""
import unittest
import numpy as np, cv2
from tt_scout.heads import find_heads, plate_of, thresholds, standing_zone, HEAD_H, HEAD_SPAN
from tt_scout.table import Table
from tt_scout.annotate import map_compact_height, map_sprite, map_frame, tag_spot, TAG_H, MAP_FULL_BOTTOM, MAP_MIN_H, MAP_H

TOP, NET = 300, 480                     # table's far edge and the net, in a 960x540 frame


def hall(dark=False):
    f = np.full((540, 960, 3), (40, 30, 25) if dark else (170, 70, 40), np.uint8)       # backdrop: dark wall, or a blue curtain
    f[TOP - 60:TOP - 20] = (20, 20, 20)                                                  # a black banner across the hall
    f[TOP + 40:] = (40, 150, 200)                                                        # floor
    cv2.fillConvexPoly(f, np.array([(200, TOP), (760, TOP), (800, TOP + 90), (160, TOP + 90)], np.int32), (160, 60, 20))   # table
    return f


def player(f, x, head_y, shirt=(30, 30, 200), hair=(60, 80, 120), arm_up=False, legs_hidden_by_table=False):
    cv2.circle(f, (x, head_y), 15, hair, -1)                                             # head, 30 px
    cv2.rectangle(f, (x - 28, head_y + 16), (x + 28, head_y + 110), shirt, -1)           # torso
    if not legs_hidden_by_table:
        cv2.rectangle(f, (x - 22, head_y + 110), (x + 22, head_y + 230), (30, 30, 30), -1)
    if arm_up:
        cv2.rectangle(f, (x + 20, head_y - 70), (x + 29, head_y + 20), shirt, -1)        # a raised arm, 9 px wide, above the head
    return f


class Heads(unittest.TestCase):
    def check(self, plate, frame, want, tol_x=6):
        got = find_heads(frame, plate, TOP, NET, thr=thresholds(plate))
        for end, w in want.items():
            if w is None:
                self.assertIsNone(got[end], end)
            else:
                self.assertIsNotNone(got[end], end)
                x, y, sure = got[end]
                self.assertTrue(sure, end); self.assertLess(abs(x - w[0]), tol_x, (end, x)); self.assertLess(abs(y - w[1]), 8, (end, y))

    def test_heads_of_both_players_and_nothing_in_an_empty_hall(self):
        plate = hall()
        self.check(plate, plate.copy(), {"near": None, "far": None})
        f = player(player(hall(), 120, TOP - 100), 850, TOP - 90, shirt=(20, 20, 20))    # far player in black, crossing the black banner
        self.check(plate, f, {"near": (120, TOP - 100), "far": (850, TOP - 90)})

    def test_a_raised_arm_is_not_the_head_and_legs_behind_the_table_do_not_matter(self):
        plate = hall()
        f = player(hall(), 150, TOP - 95, arm_up=True)
        f = player(f, 700, TOP - 110, legs_hidden_by_table=True)
        self.check(plate, f, {"near": (150, TOP - 95), "far": (700, TOP - 110)}, tol_x=10)   # an arm touching the head pulls x a few px

    def test_dark_hair_on_a_dark_wall_is_still_found(self):
        plate = hall(dark=True)
        f = player(hall(dark=True), 200, TOP - 100, hair=(15, 12, 10), shirt=(40, 120, 230))
        self.check(plate, f, {"near": (200, TOP - 100)})

    def test_a_bystander_joined_to_the_player_does_not_take_the_balloon(self):
        plate = hall()
        f = player(hall(), 420, TOP - 100)                                               # the player, feet on the floor
        f = player(f, 180, TOP - 150, shirt=(90, 90, 90), legs_hidden_by_table=True)     # someone behind, cut off at the barrier, and taller
        cv2.rectangle(f, (180, TOP - 34), (420, TOP - 26), (90, 90, 90), -1)             # joined to them by an arm, so it is all one shape
        self.check(plate, f, {"near": (420, TOP - 100)}, tol_x=10)                       # the head over the feet, not the highest one

    def test_a_crowd_behind_the_barrier_stands_outside_the_players_zone(self):
        table = Table([[775.9, 485.1], [423.1, 503.3], [873.0, 595.4], [1285.7, 552.3]])   # a hall with an audience, 1920x1080
        standing = standing_zone(table, 0.5)                                               # read at 960x540
        for foot in ((375, 295), (760, 530), (302, 266), (468, 458)):                      # the two players, over four rallies
            self.assertTrue(standing(*foot), foot)
        for foot in ((15, 209), (340, 213)):                                               # a spectator at the end, a row along the side
            self.assertFalse(standing(*foot), foot)

    def test_the_plate_is_the_empty_hall_when_players_move(self):
        frames = [player(hall(), 100 + 37 * k, TOP - 100) for k in range(9)]
        self.assertLess(int(cv2.absdiff(plate_of(frames), hall()).max()), 3)


class LandingMap(unittest.TestCase):
    def test_full_until_heads_under_it_reach_its_bottom_then_just_above_them(self):
        left = 700
        self.assertEqual(map_compact_height([(800, MAP_FULL_BOTTOM + 30)] * 20, left), 0)
        self.assertEqual(map_compact_height([(300, 60)] * 20 + [(800, MAP_FULL_BOTTOM + 30)] * 5, left), 0)     # high heads, but not under it
        self.assertEqual(map_compact_height([(800, 150)] * 1 + [(800, 220)] * 30, left), 0)                     # one stretch is not enough
        th = map_compact_height([(820, 120)] * 20, left)
        self.assertTrue(MAP_MIN_H <= th < MAP_H); self.assertLessEqual(10 + th + 16 + 8, 120 - 6)                # compact panel ends above them
        self.assertEqual(map_compact_height([(820, 40)] * 20, left), MAP_MIN_H)
        compact = map_sprite("A", "B", th)
        self.assertLessEqual(compact.shape[0], th + 16 + 10); self.assertEqual(map_frame(th)[3], th)


class BalloonsAvoidHeads(unittest.TestCase):
    def test_a_slot_that_would_touch_the_other_players_head_is_not_used(self):
        w = 64
        free = tag_spot((400, 250), -1, w, [], 960)
        self.assertEqual((free[0], free[3]), (0, "right"))                                   # left of the head by default
        other = [(free[1] - 20, 250 - 22, free[1] + 20, 250 + 20)]                          # another head sits exactly there
        moved = tag_spot((400, 250), -1, w, [], 960, avoid=other)
        box = (moved[1] - w / 2 - 13, moved[2] - TAG_H / 2, moved[1] + w / 2 + 13, moved[2] + TAG_H / 2 + 13)
        self.assertNotEqual(moved[0], 0)
        self.assertEqual(max(0, min(box[2], other[0][2]) - max(box[0], other[0][0])) * max(0, min(box[3], other[0][3]) - max(box[1], other[0][1])), 0)


if __name__ == "__main__":
    unittest.main()
