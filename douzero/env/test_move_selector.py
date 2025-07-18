import unittest
from douzero.env.move_selector import *
from douzero.env.utils import *

class TestMoveSelector(unittest.TestCase):
    def setUp(self):
        # Define mastercard lists for testing
        self.mastercard_list1 = [3, 7]  # Basic mastercards
        self.mastercard_list2 = [5, 9]  # Different mastercards
        self.mastercard_list3 = [3, 3]  # Duplicate mastercards
        self.mastercard_list4 = []      # No mastercards

    def test_common_handle(self):
        # Test basic comparison
        moves = [[4, 4], [5, 5], [6, 6]]
        rival_move = [4, 4]
        result = common_handle(moves, rival_move)
        self.assertEqual(result, [[5, 5], [6, 6]])

        # Test with mastercards
        moves = [[4, 3], [5, 7], [6, 3]]
        rival_move = [4, 3]
        result = common_handle(moves, rival_move)
        self.assertEqual(result, [[5, 7], [6, 3]])

    def test_common_handle_serial(self):
        # Test serial singles
        moves = [[3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]]
        rival_move = [3, 4, 5, 6]
        result = common_handle_serial(moves, rival_move, self.mastercard_list1)
        self.assertEqual(result, [[4, 5, 6, 7], [5, 6, 7, 8]])

        # Test with mastercards
        moves = [[3, 4, 5, 7], [4, 5, 6, 3], [5, 6, 7, 3]]
        rival_move = [3, 4, 5, 7]
        result = common_handle_serial(moves, rival_move, self.mastercard_list1)
        self.assertEqual(result, [[4, 5, 6, 3], [5, 6, 7, 3]])

    def test_common_handle_triple_and_pair(self):
        # Test triples
        moves = [[4, 4, 4], [5, 5, 5], [6, 6, 6]]
        rival_move = [4, 4, 4]
        result = common_handle_triple_and_pair(moves, rival_move, self.mastercard_list1)
        self.assertEqual(result, [[5, 5, 5], [6, 6, 6]])

        # Test with mastercards
        moves = [[4, 4, 3], [5, 5, 7], [6, 6, 3]]
        rival_move = [4, 4, 3]
        result = common_handle_triple_and_pair(moves, rival_move, self.mastercard_list1)
        self.assertEqual(result, [[5, 5, 7], [6, 6, 3]])

    def test_common_handle_bomb(self):
        # Test regular bombs
        moves = [[4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]]
        rival_move = [4, 4, 4, 4]
        result = common_handle_bomb(moves, rival_move, self.mastercard_list1)
        self.assertEqual(result, [[5, 5, 5, 5], [6, 6, 6, 6]])

        # Test with mastercards
        moves = [[4, 4, 4, 3], [5, 5, 5, 5], [6, 6, 6, 3]]
        rival_move = [8,8,8,3]
        result = common_handle_bomb(moves, rival_move, self.mastercard_list1)
        self.assertEqual(result, [[5, 5, 5, 5]])

        # # Test king bomb
        # moves = [[20, 30], [4, 4, 4, 4]]
        # rival_move = [4, 4, 4, 4]
        # result = common_handle_bomb(moves, rival_move, self.mastercard_list1)
        # self.assertEqual(result, [[20, 30]])

    def test_filter_type_1_single(self):
        # Test single card filtering
        moves = [[4], [7], [13], [14]]  # Available moves
        rival_move = [3]  # Rival's move
        filtered = filter_type_1_single(moves, rival_move, self.mastercard_list1)
        self.assertEqual(filtered, [[4], [7], [13], [14]])

        # Test when some moves can't beat rival
        rival_move = [8]
        filtered = filter_type_1_single(moves, rival_move, self.mastercard_list1)
        self.assertEqual(filtered, [[13], [14]])

    def test_filter_type_2_pair(self):
        # Test pair filtering
        moves = [[3, 3], [7, 7], [13, 13]]
        rival_move = [2, 2]
        filtered = filter_type_2_pair(moves, rival_move, self.mastercard_list1)
        self.assertEqual(filtered, [[3, 3], [7, 7], [13, 13]])

        rival_move = [8, 8]
        filtered = filter_type_2_pair(moves, rival_move, self.mastercard_list1)
        self.assertEqual(filtered, [[13, 13]])

    def test_filter_type_3_triple(self):
        moves = [[4, 4, 4], [5, 5, 5], [6, 6, 6], [4, 4, 3], [5, 5, 7]]
        rival_move = [4, 4, 4]
        result = filter_type_3_triple(moves, rival_move, self.mastercard_list1)
        self.assertEqual(result, [[5, 5, 5], [6, 6, 6], [5, 5, 7]])

    def test_filter_type_4_bomb(self):
        # Test regular bomb
        moves = [[4, 4, 4, 4], [8, 8, 8, 8], [13, 13, 13, 13]]
        rival_move = [3, 3, 3, 3]
        filtered = filter_type_4_bomb(moves, rival_move, self.mastercard_list1)
        self.assertEqual(filtered, [])

        # Test soft bomb vs hard bomb
        moves = [[4, 4, 4, 13], [8, 8, 8, 7]]
        rival_move = [5,5,5,5]
        filtered = filter_type_4_bomb(moves, rival_move, self.mastercard_list1)
        self.assertEqual(filtered, [])

    def test_filter_type_6_3_1(self):
        # Test 3+1 combination
        moves = [[3, 3, 3, 5], [7, 7, 7, 8], [13, 13, 13, 4]]
        rival_move = [6, 6, 6, 4]
        filtered = filter_type_6_3_1(moves, rival_move, self.mastercard_list1)
        self.assertEqual(filtered, [[3, 3, 3, 5], [7, 7, 7, 8], [13, 13, 13, 4]])

        rival_move = [8, 8, 8, 4]
        filtered = filter_type_6_3_1(moves, rival_move, self.mastercard_list1)
        self.assertEqual(filtered, [[13, 13, 13, 4]])

    def test_filter_type_7_3_2(self):
        moves = [[4, 4, 4, 5, 5], [5, 5, 5, 6, 6], [6, 6, 6, 7, 7], [4, 4, 4, 3, 3], [5, 5, 5, 7, 7]]
        rival_move = [4, 4, 4, 5, 5]
        result = filter_type_7_3_2(moves, rival_move, self.mastercard_list1)
        self.assertEqual(result, [[5, 5, 5, 6, 6], [6, 6, 6, 7, 7], [5, 5, 5, 7, 7]])

    def test_filter_type_8_serial_single(self):
        # Test serial single
        moves = [[3, 4, 5], [7, 8, 9], [13, 5, 6]]
        rival_move = [2, 3, 4]
        filtered = filter_type_8_serial_single(moves, rival_move, self.mastercard_list1)
        self.assertEqual(filtered, [[3, 4, 5], [7, 8, 9]])

    def test_filter_type_9_serial_pair(self):
        moves = [[3, 3, 4, 4], [4, 4, 5, 5], [5, 5, 6, 6], [3, 3, 4, 7], [4, 4, 5, 3]]
        rival_move = [3, 3, 4, 4]
        result = filter_type_9_serial_pair(moves, rival_move, self.mastercard_list1)
        self.assertEqual(result, [[4, 4, 5, 5], [5, 5, 6, 6], [4, 4, 5, 3]])

    def test_filter_type_10_serial_triple(self):
        moves = [[3, 3, 3, 4, 4, 4], [4, 4, 4, 5, 5, 5], [5, 5, 5, 6, 6, 6], 
                [3, 3, 3, 4, 4, 7], [4, 4, 4, 5, 5, 3]]
        rival_move = [3, 3, 3, 4, 4, 4]
        result = filter_type_10_serial_triple(moves, rival_move, self.mastercard_list1)
        self.assertEqual(result, [[4, 4, 4, 5, 5, 5], [5, 5, 5, 6, 6, 6], [4, 4, 4, 5, 5, 3]])

    def test_filter_type_11_serial_3_1(self):
        # Test serial 3+1
        moves = [
            [3, 3, 3, 4, 5, 5, 5, 6],  # Two groups of 3+1
            [7, 7, 7, 8, 9, 9, 9, 10],
            [13, 13, 13, 4, 14, 14, 14, 6]
        ]
        rival_move = [2, 2, 2, 4, 3, 3, 3, 5]
        filtered = filter_type_11_serial_3_1(moves, rival_move, self.mastercard_list1)
        self.assertEqual(filtered, [
            [3, 3, 3, 4, 5, 5, 5, 6],
            [7, 7, 7, 8, 9, 9, 9, 10],
            [13, 13, 13, 4, 14, 14, 14, 6]
        ])

    def test_filter_type_12_serial_3_2(self):
        moves = [[3, 3, 3, 4, 4, 4, 5, 5, 6, 6], [4, 4, 4, 5, 5, 5, 6, 6, 7, 7],
                [5, 5, 5, 6, 6, 6, 7, 7, 8, 8], [3, 3, 3, 4, 4, 4, 5, 5, 7, 7]]
        rival_move = [3, 3, 3, 4, 4, 4, 5, 5, 6, 6]
        result = filter_type_12_serial_3_2(moves, rival_move, self.mastercard_list1)
        self.assertEqual(result, [  [5, 5, 5, 6, 6, 6, 7, 7, 8, 8]])

    def test_filter_type_13_4_2(self):
        moves = [[4, 4, 4, 4, 5, 6], [5, 5, 5, 5, 6, 7], [6, 6, 6, 6, 7, 8],
                [4, 4, 4, 4, 5, 3], [5, 5, 5, 5, 6, 7]]
        rival_move = [4, 4, 4, 4, 5, 6]
        result = filter_type_13_4_2(moves, rival_move, self.mastercard_list1)
        self.assertEqual(result, [[5, 5, 5, 5, 6, 7], [6, 6, 6, 6, 7, 8], [5, 5, 5, 5, 6, 7]])

    def test_filter_type_14_4_22(self):
        moves = [[4, 4, 4, 4, 5, 5, 6, 6], [5, 5, 5, 5, 6, 6, 7, 7],
                [6, 6, 6, 6, 7, 7, 8, 8], [4, 4, 4, 4, 5, 5, 3, 3]]
        rival_move = [4, 4, 4, 4, 5, 5, 6, 6]
        result = filter_type_14_4_22(moves, rival_move, self.mastercard_list1)
        self.assertEqual(result, [[5, 5, 5, 5, 6, 6, 7, 7], 
                                [6, 6, 6, 6, 7, 7, 8, 8]])

    def test_king_bomb(self):
        # Test king bomb (special case)
        moves = [[20, 30]]  # King bomb
        rival_move = [8, 8, 8, 8]  # Regular bomb
        filtered = filter_type_4_bomb(moves, rival_move, self.mastercard_list1)
        self.assertEqual(filtered, [[20, 30]])

if __name__ == '__main__':
    unittest.main() 