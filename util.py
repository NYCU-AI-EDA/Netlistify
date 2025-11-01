type_need_orientation = ['pmos','nmos','pnp','npn','voltage'
        ,'current','diode','and','or','xor','not','op','tgate']

def extend_box(pos):
    x1, y1, x2, y2 = pos
    return [x1-3, y1, x2+3, y2]

def cut_box(config, type, pos, orientation):
    x1, y1, x2, y2 = pos
    mid_x = x1 + (x2 - x1) // 2
    mid_y = y1 + (y2 - y1) // 2
    one_third_y = y1 + (y2 - y1) // 3
    two_third_y = y1 + 2 * (y2 - y1) // 3
    one_third_x = x1 + (x2 - x1) // 3
    two_third_x = x1 + 2 * (x2 - x1) // 3
    one_fourth_y = y1 + (y2 - y1) // 4
    three_fourth_y = y1 + 3 * (y2 - y1) // 4
    one_fourth_x = x1 + (x2 - x1) // 4
    three_fourth_x = x1 + 3 * (x2 - x1) // 4

    def split_horizontal_vertical():
        if orientation in ['R0', 'R180']:
            # left_box = [x1, y1, mid_x, y2]
            left_box = [x1, y1, x1 + (x2 - x1) // 4, y2]
            right_box = [mid_x, y1, x2, y2]
            return (left_box, right_box) if orientation == 'R0' else (right_box, left_box)
        elif orientation in ['R90', 'R270']:
            top_box = [x1, y1, x2, mid_y]
            down_box = [x1, mid_y, x2, y2]
            return (top_box, down_box) if orientation == 'R90' else (down_box, top_box)

    def split_vertical_horizontal():
        if orientation in ['R0', 'R180']:
            top_box = [x1, y1, x2, mid_y]
            down_box = [x1, mid_y, x2, y2]
            return (top_box, down_box) if orientation == 'R0' else (down_box, top_box)
        elif orientation in ['R90', 'R270']:
            left_box = [x1, y1, mid_x, y2]
            right_box = [mid_x, y1, x2, y2]
            return (left_box, right_box) if orientation == 'R90' else (right_box, left_box)

    def split_rcl():
        if abs(x2 - x1) > abs(y2 - y1):
            left_box = [x1, y1, mid_x, y2]
            right_box = [mid_x, y1, x2, y2]
            return (left_box, right_box)
        else:
            top_box = [x1, y1, x2, mid_y]
            down_box = [x1, mid_y, x2, y2]
            return (top_box, down_box)

    def split_real_c():
        if abs(x2 - x1) < abs(y2 - y1):
            left_box = [x1, y1, mid_x, y2]
            right_box = [mid_x, y1, x2, y2]
            return (left_box, right_box)
        else:
            top_box = [x1, y1, x2, mid_y]
            down_box = [x1, mid_y, x2, y2]
            return (top_box, down_box)

    def split_bjt():
        if orientation == 'R0':
            c_box = [mid_x, y1, x2, mid_y]
            b_box = [x1, y1, mid_x, y2]
            e_box = [mid_x, mid_y, x2, y2]
        elif orientation == 'R90':
            c_box = [x1, y1, mid_x, mid_y]
            b_box = [x1, mid_y, x2, y2]
            e_box = [mid_x, y1, x2, mid_y]
        elif orientation == 'R180':
            c_box = [x1, mid_y, mid_x, y2]
            b_box = [mid_x, y1, x2, y2]
            e_box = [x1, y1, mid_x, mid_y]
        elif orientation == 'R270':
            c_box = [mid_x, mid_y, x2, y2]
            b_box = [x1, y1, x2, mid_y]
            e_box = [x1, mid_y, mid_x, y2]
        elif orientation == 'MY':
            c_box = [x1, y1, mid_x, mid_y]
            b_box = [mid_x, y1, x2, y2]
            e_box = [x1, mid_y, mid_x, y2]
        elif orientation == 'MYR90':
            c_box = [x1, mid_y, mid_x, y2]
            b_box = [x1, y1, x2, mid_y]
            e_box = [mid_x, mid_y, x2, y2]
        elif orientation == 'MX':
            c_box = [mid_x, mid_y, x2, y2]
            b_box = [x1, y1, mid_x, y2]
            e_box = [mid_x, y1, x2, mid_y]
        elif orientation == 'MXR90':
            c_box = [mid_x, y1, x2, mid_y]
            b_box = [x1, mid_y, x2, y2]
            e_box = [x1, y1, mid_x, mid_y]
        return (c_box, b_box, e_box) if type == 'npn' else (e_box, b_box, c_box)

    def split_mos():
        if orientation == 'R0':
            d_box = [mid_x, y1, x2, one_third_y]
            g_box = [x1, y1, mid_x, y2]
            s_box = [mid_x, two_third_y, x2, y2]
            b_box = [mid_x, one_third_y, x2 , two_third_y]
        elif orientation == 'R90':
            d_box = [x1, y1, one_third_x, mid_y]
            g_box = [x1, mid_y, x2, y2]
            s_box = [two_third_x, y1, x2, mid_y]
            b_box = [one_third_x, y1, two_third_x, mid_y]
        elif orientation == 'R180':
            d_box = [x1, two_third_y, mid_x, y2]
            g_box = [mid_x, y1, x2, y2]
            s_box = [x1, y1, mid_x, one_third_y]
            b_box = [x1, one_third_y, mid_x, two_third_y]
        elif orientation == 'R270':
            d_box = [two_third_x, mid_y, x2, y2]
            g_box = [x1, y1, x2, mid_y]
            s_box = [x1, mid_y, one_third_x, y2]
            b_box = [one_third_x, mid_y, two_third_x, y2]
        elif orientation == 'MY':
            d_box = [x1, y1, mid_x, one_third_y]
            g_box = [mid_x, y1, x2, y2]
            s_box = [x1, two_third_y, mid_x, y2]
            b_box = [x1, one_third_y, mid_x , two_third_y]
        elif orientation == 'MYR90':
            d_box = [x1, mid_y, one_third_x, y2]
            g_box = [x1, y1, x2, mid_y]
            s_box = [two_third_x, mid_y, x2, y2]
            b_box = [one_third_x, mid_y, two_third_x, y2]
        elif orientation == 'MX':
            d_box = [mid_x, two_third_y, x2, y2]
            g_box = [x1, y1, mid_x, y2]
            s_box = [mid_x, y1, x2, one_third_y]
            b_box = [mid_x, one_third_y, x2 , two_third_y]
        elif orientation == 'MXR90':
            d_box = [two_third_x, y1, x2, mid_y]
            g_box = [x1, mid_y, x2, y2]
            s_box = [x1, y1, one_third_x, mid_y]
            b_box = [one_third_x, y1, two_third_x, mid_y]
        if config:
            return (d_box, g_box, s_box, b_box) if type == 'nmos' else (s_box, g_box, d_box, b_box)
        else:
            return (d_box, g_box, s_box, s_box) if type == 'nmos' else (s_box, g_box, d_box, d_box)


    def split_bjt_xy():
        if orientation == 'R0':
            c_box = [two_third_x, y1, x2, one_third_y]
            b_box = [x1, one_third_y, one_third_x, two_third_y]
            e_box = [two_third_x, two_third_y, x2, y2]
        elif orientation == 'R90':
            c_box = [x1, y1, one_third_x, one_third_y]
            b_box = [one_third_x, two_third_y,two_third_x , y2]
            e_box = [two_third_x, y1, x2, one_third_y]
        elif orientation == 'R180':
            c_box = [x1, two_third_y, two_third_x, y2]
            b_box = [two_third_x, one_third_y, x2, two_third_y]
            e_box = [x1, y1, one_third_x, one_third_y]
        elif orientation == 'R270':
            c_box = [two_third_x, two_third_y, x2, y2]
            b_box = [one_third_x, y1, two_third_x, one_third_y]
            e_box = [x1, two_third_y, one_third_x, y2]
        elif orientation == 'MY':
            c_box = [x1, y1, one_third_x, one_third_y]
            b_box = [two_third_x, one_third_y, x2, two_third_y]
            e_box = [x1, two_third_y, one_third_x, y2]
        elif orientation == 'MYR90':
            c_box = [x1, one_third_y, one_third_x, y2]
            b_box = [one_third_x, y1, two_third_x, one_third_y]
            e_box = [two_third_x, two_third_y, x2, y2]
        elif orientation == 'MX':
            c_box = [two_third_x, two_third_y, x2, y2]
            b_box = [x1, one_third_y, one_third_x, two_third_y]
            e_box = [two_third_x, y1, x2, one_third_y]
        elif orientation == 'MXR90':
            c_box = [two_third_x, y1, x2, one_third_y]
            b_box = [one_third_x, two_third_y, two_third_x, y2]
            e_box = [x1, y1, one_third_x, one_third_y]
        return (c_box, b_box, e_box) if type == 'npn' else (e_box, b_box, c_box)

    def split_mos_xy():
        if orientation == 'R0':
            d_box = [three_fourth_x, y1, x2, one_fourth_y]
            g_box = [x1-5, one_third_y, x1, two_third_y]
            s_box = [three_fourth_x, three_fourth_y, x2, y2]
            b_box = [three_fourth_x, one_third_y, x2 , two_third_y]
        elif orientation == 'R90':
            d_box = [x1, y1, one_fourth_x, one_fourth_y]
            g_box = [one_third_x, y2, two_third_x, y2+5]
            s_box = [three_fourth_x, y1, x2, one_fourth_y]
            b_box = [one_third_x, y1, two_third_x, mid_y]
        elif orientation == 'R180':
            d_box = [x1, three_fourth_y, one_fourth_x, y2]
            g_box = [x2, one_third_y, x2+5, two_third_y]
            s_box = [x1, y1, one_fourth_x, one_fourth_y]
            b_box = [x1, one_third_y, mid_x, two_third_y]
        elif orientation == 'R270':
            d_box = [three_fourth_x, three_fourth_y, x2, y2]
            g_box = [one_third_x, y1-5, two_third_x, y1]
            s_box = [x1, three_fourth_y, one_fourth_x, y2]
            b_box = [one_third_x, mid_y, two_third_x, y2]
        elif orientation == 'MY':
            d_box = [x1, y1, one_fourth_x, one_fourth_y]
            g_box = [x2, one_third_y, x2+5, two_third_y]
            s_box = [x1, three_fourth_y, one_fourth_x, y2]
            b_box = [x1, one_third_y, mid_x , two_third_y]
        elif orientation == 'MYR90':
            d_box = [x1, three_fourth_y, one_fourth_x, y2]
            g_box = [one_third_x, y1-5, two_third_x, y1]
            s_box = [three_fourth_x, three_fourth_y, x2, y2]
            b_box = [one_third_x, mid_y, two_third_x, y2]
        elif orientation == 'MX':
            d_box = [three_fourth_x, three_fourth_y, x2, y2]
            g_box = [x1-5, one_third_y, x1, two_third_y]
            s_box = [three_fourth_x, y1, x2, one_fourth_y]
            b_box = [mid_x, one_third_y, x2 , two_third_y]
        elif orientation == 'MXR90':
            d_box = [three_fourth_x, y1, x2, one_fourth_y]
            g_box = [one_third_x, y2, two_third_x, y2+5]
            s_box = [x1, y1, one_fourth_x, one_fourth_y]
            b_box = [one_third_x, y1, two_third_x, mid_y]
        if config:
            return (d_box, g_box, s_box, b_box) if type == 'nmos' else (s_box, g_box, d_box, b_box)
        else:
            return (d_box, g_box, s_box, s_box) if type == 'nmos' else (s_box, g_box, d_box, d_box)

    def split_tgate():
        if orientation == 'R0':
            left_box = [x1, one_third_y, one_third_x, two_third_y] #藍
            right_box = [two_third_x, one_third_y, x2, two_third_y] #橘
            down_box = [one_third_x, two_third_y, two_third_x, y2] #綠
            up_box = [one_third_x, y1, two_third_x, one_third_y] #黃
        elif orientation == 'R90':
            left_box = [one_third_x, two_third_y, two_third_x, y2]
            right_box = [one_third_x, y1, two_third_x, one_third_y]
            down_box = [one_third_x, one_third_y, two_third_x, two_third_y]
            up_box = [x1, one_third_y, one_third_x, two_third_y]
        elif orientation == 'R180':
            left_box = [two_third_x, one_third_y, x2, two_third_y]
            right_box = [x1, one_third_y, one_third_x, two_third_y]
            down_box = [one_third_x, y1, two_third_x, one_third_y]
            up_box = [one_third_x, two_third_y, two_third_x, y2]
        elif orientation == 'R270':
            left_box = [one_third_x, y1, two_third_x, one_third_y]
            right_box = [one_third_x, two_third_y, two_third_x, y2]
            down_box = [x1, one_third_y, one_third_x, two_third_y]
            up_box = [two_third_x, one_third_y, x2, two_third_y]
        return (left_box, right_box, down_box, up_box)

    def split_op():
        if orientation == 'R0':
            pos_box = [x1, y1, mid_x, mid_y]
            neg_box = [x1, mid_y, mid_x, y2]
            output_box = [mid_x, y1, x2, y2]
        elif orientation == 'R90':
            pos_box = [x1, mid_y, mid_x, y2]
            neg_box = [mid_x, mid_y, x2, y2]
            output_box = [x1, y1, x2, mid_y]
        elif orientation == 'R180':
            pos_box = [mid_x, mid_y, x2, y2]
            neg_box = [mid_x, y1, x2, mid_y]
            output_box = [x1, y1, mid_x, y2]
        elif orientation == 'R270':
            pos_box = [mid_x, y1, x2, mid_y]
            neg_box = [x1, y1, mid_x, mid_y]
            output_box = [mid_x, mid_y, x2, y2]
        elif orientation == 'MX':
            pos_box = [x1, mid_y, mid_x, y2]
            neg_box = [x1, y1, mid_x, mid_y]
            output_box = [mid_x, y1, x2, y2]
        elif orientation == 'MY':
            pos_box = [mid_x, y1, x2, mid_y]
            neg_box = [mid_x, mid_y, x2, y2]
            output_box = [x1, y1, mid_x, y2]
        elif orientation == 'MXR90':
            pos_box = [mid_x, mid_y, x2, y2]
            neg_box = [x1, mid_y, mid_x, y2]
            output_box = [x1, y1, x2, mid_y]
        elif orientation == 'MYR90':
            pos_box = [x1, y1, mid_x, mid_y]
            neg_box = [mid_x, y1, x2, mid_y]
            output_box = [x1, mid_y, x2, y2]
        return (pos_box, neg_box, output_box)

    if config:
        if type == 'gnd':
            return [pos]
        elif type == 'func':
            return [[pos[0]-3, pos[1], pos[2]+3, pos[3]]]
        elif type in ['resistor', 'capacity', 'inductor']:
        # elif type in ['resistor', 'inductor']:
            return split_rcl()
        # elif type == 'capacity':
        #     return split_real_c()
        elif type in ['or', 'xor', 'and', 'not']:
            return split_horizontal_vertical()
        elif type in ['voltage', 'current', 'diode']:
            return split_vertical_horizontal()
        elif type in ['npn', 'pnp']:
            return split_bjt()
        elif type in ['pmos', 'nmos']:
            return split_mos()
        elif type == 'tgate':
            return split_tgate()
        elif type == 'op':
            return split_op()
    else:
        if type == 'gnd':
            return [pos]
        elif type == 'func':
            return [[pos[0]-3, pos[1], pos[2]+3, pos[3]]]
        # elif type in ['resistor', 'capacity', 'inductor']:
        elif type in ['resistor', 'inductor']:
            return split_rcl()
        elif type == 'capacity':
            return split_real_c()
        elif type in ['or', 'xor', 'and', 'not']:
            return split_horizontal_vertical()
        elif type in ['voltage', 'current', 'diode']:
            return split_vertical_horizontal()
        elif type in ['npn', 'pnp']:
            return split_bjt_xy()
        elif type in ['pmos', 'nmos']:
            return split_mos_xy()
        elif type == 'tgate':
            return split_tgate()
        elif type == 'op':
            return split_op()
