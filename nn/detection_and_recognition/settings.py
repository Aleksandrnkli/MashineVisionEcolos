def get_const_for_MTCNN(learning_mode):
    if learning_mode == "LPR":
        image_size = (94, 24)
        mini_lp_size = (50, 15)
        cls_num = 2
        kernel = (3, 5)
        mp = (2, 5)
        width = 47
        height = 12
        linear = 2 * 5
    elif learning_mode == "CAR":
        image_size = (24, 24)
        mini_lp_size = (15, 15)
        cls_num = 6
        kernel = (3, 3)
        mp = (2, 2)
        width = 12
        height = 12
        linear = 1
    else:
        raise ValueError(f"Learmimg mode {learning_mode} is not supported.")
    return image_size, width, height, mini_lp_size, cls_num, kernel, mp, linear

def get_const_for_execute_mtcnn_net(learning_mode):
    if learning_mode == "LPR":
        stride = (2, 5)
        cell_size = (12, 44)
        size = (94, 24)
    elif learning_mode == "CAR":
        stride = (2, 2)
        cell_size = (12, 12)
        size = (24, 24)
    else:
        raise ValueError(f"Learmimg mode {learning_mode} is not supported.")
    return stride, cell_size, size
