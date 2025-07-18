from douzero.env.move_detector import get_move_type

def test_get_move_type():
    mastercard_list = [6,7]

    # move = [6,6,7,6]
    # print(get_move_type(move,mastercard_list))
    # move = [6,8,9,3]
    # print(get_move_type(move,mastercard_list))
    # move = [6,8,9,7]
    # print(get_move_type(move,mastercard_list))
    # move = [6,8,8,6]
    # print(get_move_type(move,mastercard_list))
    # move = [9,8,8,9]
    # print(get_move_type(move,mastercard_list))
    # move = [9,8,8,8]
    # print(get_move_type(move,mastercard_list))

    # move = [3, 3, 4, 4, 5, 5]
    # print(move,get_move_type(move,mastercard_list))

    # move = [3,3,4]
    # print(move,get_move_type(move,mastercard_list))


    # move = [4,4,5,6,7]
    # print(move,get_move_type(move,mastercard_list))
    # move = [6,6,7,12,13]
    # print(move,get_move_type(move,mastercard_list))
    # move = [8,9,10,11,12]
    # print(move,get_move_type(move,mastercard_list))
    # move = [6,6,6,7,7]
    # print(move,get_move_type(move,mastercard_list))

    # move = [8,8,8,8,12,12]
    # print(get_move_type(move,mastercard_list))
    # move = [5,5,7,8,9,10]
    # print(move,get_move_type(move,mastercard_list))
    # move = [6,6,7,6,6,7]
    # print(move,get_move_type(move,mastercard_list))
    # move = [6,6,8,8,9,9]
    # print(move,get_move_type(move,mastercard_list))
    # move = [6,8,8,9,9,10]
    # print(move,get_move_type(move,mastercard_list))

    # move = [3,4,5,6,6,7]
    # print(move,get_move_type(move,mastercard_list))
    # move = [6,6,7,9,9,10]
    # print(move,get_move_type(move,mastercard_list))

    # move = [8,10,6,6,7,6,6,7,7]
    # print(move,get_move_type(move,mastercard_list))

    # move = [3,3,4,4,5,5,6,6,7,5]
    # print(move,get_move_type(move,mastercard_list))

    move = [6, 6, 11, 11, 11, 12, 12, 12, 13, 13]
    move =[3, 7, 12, 4, 4, 4, 5, 5, 12, 6, 8, 11]
    mastercard_list = [7,12]

    from douzero.env.move_generator_for_detector import MovesGener
    mg= MovesGener(move,mastercard_list)
    print(mg.gen_type_11_serial_3_1(repeat_num=len(move)//4))
    print(move,get_move_type(move,mastercard_list))



test_get_move_type()