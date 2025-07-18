import unittest
from douzero.env.move_detector import get_move_type

class TestMoveDetector(unittest.TestCase):
    def setUp(self):
        # Define some mastercard lists for testing
        self.mastercard_list1 = [3, 7]  # Basic mastercards
        self.mastercard_list2 = [5, 9]  # Different mastercards
        self.mastercard_list3 = [6, 7]  # Duplicate mastercards
        self.mastercard_list4 = [14, 12]      # No mastercards

    def test_single_cards(self):
        # Test single regular cards
        self.assertEqual(get_move_type([3], self.mastercard_list1)['type'], 1)  # TYPE_1_SINGLE
        self.assertEqual(get_move_type([7], self.mastercard_list1)['type'], 1)  # TYPE_1_SINGLE
        
        # Test single mastercard
        self.assertEqual(get_move_type([3], self.mastercard_list1)['type'], 1)  # Should still be single
        
        # Test jokers
        self.assertEqual(get_move_type([20], self.mastercard_list1)['type'], 1)  # Single joker
        self.assertEqual(get_move_type([30], self.mastercard_list1)['type'], 1)  # Single joker

    def test_pairs(self):
        # Test regular pairs
        self.assertEqual(get_move_type([4, 4], self.mastercard_list1)['type'], 2)  # TYPE_2_PAIR
        
        # Test pairs with one mastercard
        self.assertEqual(get_move_type([4, 3], self.mastercard_list1)['type'], 2)  # Pair with mastercard
        
        # Test pairs with two different mastercards
        self.assertEqual(get_move_type([3, 7], self.mastercard_list1)['type'], 2)  # Two mastercards as pair
        
        # Test king bomb
        self.assertEqual(get_move_type([20, 30], self.mastercard_list1)['type'], 5)  # TYPE_5_KING_BOMB

    def test_triples(self):
        # Test regular triples
        self.assertEqual(get_move_type([4, 4, 4], self.mastercard_list1)['type'], 3)  # TYPE_3_TRIPLE
        
        # Test triples with one mastercard
        self.assertEqual(get_move_type([4, 4, 3], self.mastercard_list1)['type'], 3)  # Triple with mastercard
        
        # Test triples with two mastercards
        self.assertEqual(get_move_type([4, 3, 7], self.mastercard_list1)['type'], 3)  # Triple with two mastercards

    def test_bombs(self):
        # Test regular bombs
        self.assertEqual(get_move_type([4, 4, 4, 4], self.mastercard_list1)['type'], 4)  # TYPE_4_BOMB
        
        # Test bombs with mastercards
        self.assertEqual(get_move_type([4, 4, 4, 3], self.mastercard_list1)['type'], 4)  # Bomb with mastercard
        
        # Test bombs with multiple mastercards
        self.assertEqual(get_move_type([4, 4, 3, 7], self.mastercard_list1)['type'], 4)  # Bomb with two mastercards

    def test_serial_singles(self):
        # Test regular serial singles
        self.assertEqual(get_move_type([3, 4, 5, 6, 7], self.mastercard_list1)['type'], 8)  # TYPE_8_SERIAL_SINGLE
            
        # Test serial singles with multiple mastercards
        self.assertEqual(get_move_type([3, 4, 7, 3], self.mastercard_list1)['type'], 8)  # Serial with two mastercards

    def test_serial_pairs(self):
        # Test regular serial pairs
        self.assertEqual(get_move_type([3, 3, 4, 4, 5, 5], self.mastercard_list1)['type'], 9)  # TYPE_9_SERIAL_PAIR
        
        # Test serial pairs with mastercards
        self.assertEqual(get_move_type([3, 3, 4, 7], self.mastercard_list1)['type'], 9)  # Serial pair with mastercard

    def test_serial_triples(self):
        # Test regular serial triples
        self.assertEqual(get_move_type([3, 3, 3, 4, 4, 4], self.mastercard_list1)['type'], 10)  # TYPE_10_SERIAL_TRIPLE
        
        # Test serial triples with mastercards
        self.assertEqual(get_move_type([3, 3, 3, 4, 4, 7], self.mastercard_list1)['type'], 10)  # Serial triple with mastercard

    def test_3_1_combinations(self):
        # Test regular 3+1
        self.assertEqual(get_move_type([3, 3, 3, 4], self.mastercard_list1)['type'], 6)  # TYPE_6_3_1
        
        # Test 3+1 with mastercards
        self.assertEqual(get_move_type([3, 3, 3, 7], self.mastercard_list1)['type'], 6)  # 3+1 with mastercard

    def test_3_2_combinations(self):
        # Test regular 3+2
        self.assertEqual(get_move_type([3, 3, 3, 4, 4], self.mastercard_list1)['type'], 7)  # TYPE_7_3_2
        
        # Test 3+2 with mastercards
        self.assertEqual(get_move_type([3, 3, 3, 4, 7], self.mastercard_list1)['type'], 7)  # 3+2 with mastercard

    def test_invalid_moves(self):
        # Test invalid moves
        # self.assertEqual(get_move_type([3, 4], self.mastercard_list1)['type'], 15)  # TYPE_15_WRONG
        # self.assertEqual(get_move_type([3, 3, 4], self.mastercard_list1)['type'], 15)  # Invalid triple
        self.assertEqual(get_move_type([3, 3, 4, 4, 5], self.mastercard_list3)['type'], 15)  # Invalid serial
    
    def test_4_22(self):
        self.assertEqual(get_move_type([6, 6, 11, 11, 11, 12, 12, 12, 13, 13], self.mastercard_list4)['type'], 14)  # TYPE_14_4_22

if __name__ == '__main__':
    unittest.main() 