import unittest
from douzero.env.oracle_upgrade import get_min_steps_to_win_bfs

class TestOracleMinSteps(unittest.TestCase):
    def test_no_wildcards(self):
        # No wildcard cases
        cases = [
            ([3, 3, 3], [], 1),                      # Trio -> 1 step
            ([4, 4, 5, 5, 6, 6], [], 1),             # Pair sequence 4-6 -> 1 step
            ([7, 7, 7, 7], [], 1),                   # Bomb -> 1 step
            ([8, 9, 10, 11, 12], [], 1),             # Straight -> 1 step
            ([3, 4, 5,5], [], 2),                      # Short straight -> 1 step
            ([5, 5, 6, 7, 8], [], 4),                # Pair + single straight -> 2 steps
        ]
        for hand, wcs, expected in cases:
            with self.subTest(hand=hand, wildcards=wcs):
                result = get_min_steps_to_win_bfs(hand, wcs)
                self.assertEqual(result, expected)

    def test_single_wildcard(self):
        cases = [
            ([9, 9, 10], [4], 2),                    # Use wildcard for trio
            ([3, 4, 5], [4], 2),                    # wildcard in straight
        ]
        for hand, wcs, expected in cases:
            with self.subTest(hand=hand, wildcards=wcs):
                result = get_min_steps_to_win_bfs(hand, wcs)
                self.assertEqual(result, expected)

    def test_double_wildcards(self):
        cases = [
            ([6, 6, 6, 7, 7, 7, 9, 9, 10, 10], [3, 4], 1),  # 666777 + 991010 + wildcards
            ([12, 12,11,11], [11, 12], 1),                          # wildcard-only bomb
            ([12, 12,10,9,8], [11, 12], 1),
        ]
        for hand, wcs, expected in cases:
            with self.subTest(hand=hand, wildcards=wcs):
                result = get_min_steps_to_win_bfs(hand, wcs)
                self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
