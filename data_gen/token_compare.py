import json
import math

# from PIL import Image
import cv2
import os
import ast
import numpy as np


def save_img_recursion(filepath, file_name, save_path, save_name):
    for file in os.listdir(filepath):
        full_file_path = os.path.join(filepath, file)
        if os.path.isdir(full_file_path):
            save_img_recursion(full_file_path, file_name, save_path, save_name)
        else:
            if os.path.splitext(full_file_path)[1] == '.png':
                if file_name == file:
                    try:
                        print(full_file_path, save_path + save_name + '.png')
                        # img = Image.open(full_file_path)
                        img = cv2.imread(full_file_path)

                        # img.show()
                        # img.save(save_path+save_name+'.png', 'PNG')

                        return True
                    except Exception as e:
                        print(e)
                        print(full_file_path, save_name)
                        return False


def image_resizer(img):
    max_h = 192
    max_w = 672
    h, w = img.shape

    if w > max_w:
        ratio = max_w / w
        new_h = int(h * ratio)
        img = cv2.resize(img, (max_w, new_h))
        # h, w = new_h, max_w
    h, w = img.shape

    if h > max_h:
        ratio = max_h / h
        new_w = int(w * ratio)
        img = cv2.resize(img, (new_w, max_h))
        # h, w = max_h, new_w

    h, w = img.shape

    h_remain = h % 32
    if h_remain < 16:
        new_h = h - h_remain
    else:
        new_h = h + (32 - h_remain)

    w_remain = w % 32
    if w_remain < 16:
        new_w = w - w_remain
    else:
        new_w = w + (32 - w_remain)
    if new_w < 32:
        new_w = 32
    if new_h < 32:
        new_h = 32
    img = cv2.resize(img, (new_w, new_h))

    return img


def image_add_padding(_img):
    h, w = _img.shape
    h_remain = h % 32

    if h_remain > 0:
        new_h = h + (32 - h_remain)
    else:
        new_h = h
    w_remain = w % 32
    if w_remain > 0:
        new_w = w + (32 - w_remain)
    else:
        new_w = w
    new_img = np.zeros((new_h, new_w), dtype=np.uint8)

    coordinates1 = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
    color_list = []
    for i in range(3):
        l_t = _img[coordinates1[0][0] + i, coordinates1[0][1] + i]
        color_list.append(l_t)
        r_t = _img[coordinates1[0][0] - (i + 1), coordinates1[0][1] + i]
        color_list.append(r_t)
        r_b = _img[coordinates1[0][0] - (i + 1), coordinates1[0][1] - (i + 1)]
        color_list.append(r_b)
        l_b = _img[coordinates1[0][0] + i, coordinates1[0][1] - (i + 1)]
        color_list.append(l_b)

    mean_color = np.mean(color_list)

    new_img.fill(int(mean_color))  # 모든 픽셀을 255(흰색)로 설정

    start_x = int((new_w - w) // 2)
    start_y = int((new_h - h) // 2)

    end_x = start_x + w
    end_y = start_y + h

    new_img[start_y:end_y, start_x: end_x] = _img

    return new_img


def estimate_rotation_angle(p1, p2, p3, p4):
    # 사각형의 좌표를 네 점으로 분해합니다.
    # p1, p2, p3, p4 = box

    # 중심점을 계산합니다.
    center_x = (p1[0] + p2[0] + p3[0] + p4[0]) / 4
    center_y = (p1[1] + p2[1] + p3[1] + p4[1]) / 4

    # 사각형의 각 변의 중점을 계산합니다.
    mid1_x = (p1[0] + p2[0]) / 2
    mid1_y = (p1[1] + p2[1]) / 2
    mid2_x = (p2[0] + p3[0]) / 2
    mid2_y = (p2[1] + p3[1]) / 2

    # 두 중점을 연결한 직선의 각도를 계산합니다.
    angle_rad = np.arctan2(mid2_y - mid1_y, mid2_x - mid1_x)
    angle_deg = np.degrees(angle_rad)

    return center_x, center_y, angle_deg


def find_corner_coordinates(coordinates):
    # 좌표를 x좌표를 기준으로 정렬합니다.
    coordinates.sort(key=lambda x: x[0])

    # 가장 왼쪽에 있는 두 점을 좌측 좌표로 선택합니다.
    left_coordinates = sorted(coordinates[:2], key=lambda x: x[1])

    # 가장 오른쪽에 있는 두 점을 우측 좌표로 선택합니다.
    right_coordinates = sorted(coordinates[2:], key=lambda x: x[1])

    # 좌측 상단, 좌측 하단, 우측 상단, 우측 하단 좌표를 반환합니다.
    top_left = left_coordinates[0]
    bottom_left = left_coordinates[1]
    top_right = right_coordinates[0]
    bottom_right = right_coordinates[1]

    return top_left, top_right, bottom_right, bottom_left


def rotate_image(image, angle_deg):
    # 이미지 중심을 계산합니다.
    center = tuple(np.array(image.shape[1::-1]) / 2)

    # 회전 변환 매트릭스를 생성합니다.
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # 이미지를 회전시킵니다.
    rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return rotated_image


def image_crop_rotate(img, box_str):
    box = ast.literal_eval(box_str)
    x_values = [x for x, y in box]
    y_values = [y for x, y in box]

    top_left, top_right, bottom_right, bottom_left = find_corner_coordinates(box)

    top = min(y_values)
    left = min(x_values)
    bottom = max(y_values)
    right = max(x_values)
    cropped_img = img[int(top):int(bottom), int(left):int(right)]
    center_x, center_y, rotation_angle = estimate_rotation_angle(top_left, top_right, bottom_right, bottom_left)

    # rotated_image = rotate_image(cropped_img, rotation_angle)

    return cropped_img


def save_img(file_name, save_name, box_str):
    try:
        # img = Image.open(file_name)
        img = cv2.imread(file_name)

        # cv2.imshow('test', img)
        # cv2.waitKey(0)
        # crop

        img = image_crop_rotate(img, box_str)

        w, h, c = img.shape
        if w < 11 and h < 11:
            return False

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = image_resizer(gray_img)
        cv2.imwrite(save_name, gray_img)

        return True
    except Exception as e:
        print(e)
        print(file_name, save_name)

        return False


def save_img_full(file_name, save_name):
    try:
        img = cv2.imread(file_name)


        w, h, c = img.shape
        if w < 11 and h < 11:
            return False

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = image_resizer(gray_img)
        cv2.imwrite(save_name, gray_img)

        return True
    except Exception as e:
        print(e)
        print(file_name, save_name)

        return False


def gen_img_txt_aida(save_math_txt, log_path, save_path):
    start_path = 'C:\\Users\\user\\Downloads\\AIHUB_Latex_data\\Data\\'
    # save_path = '../../AIHUB/train/'
    save_math = open(save_math_txt, 'w', encoding='utf-8')
    cnt = 0
    new_token = open(log_path, 'r', encoding='utf-8').readlines()

    for line in new_token:
        write_latex = line.split('\t')[1]
        img_name = line.split('\t')[0]
        save_img_name = save_path + '/' + str(cnt).rjust(7, "0") + '.png'
        img_temp = save_img_full(img_name, save_img_name)
        if img_temp:
            write_latex += '\n'
            save_math.write(write_latex)
            save_math.flush()
            cnt += 1

    print(cnt)


def gen_img_txt(save_math_txt, pix2tex_token_path, log_path, save_path):
    start_path = 'C:\\Users\\user\\Downloads\\AIHUB_Latex_data\\Data\\'
    # save_path = '../../AIHUB/train/'
    save_math = open(save_math_txt, 'w', encoding='utf-8')
    pre_token = open(pix2tex_token_path, 'r', encoding='utf-8')
    json_data = json.load(pre_token)

    vocab = json_data['model']['vocab']
    print(vocab)
    cnt = 0

    new_token = open(log_path, 'r', encoding='utf-8').readlines()

    for line in new_token:
        write_latex = line.split('\t')[1] #.replace('\\', '')
        latex = write_latex.split('\n')[0]
        latex_to_token = latex.split()
        temp = True
        # for token in latex_to_token:
        #     if token not in vocab:
        #         temp = False
        #         break
        if temp:
            img_name = line.split('\t')[0]
            # load_img = img_name
            box_str = line.split('\t')[2]
            save_img_name = save_path + '/' + str(cnt).rjust(7, "0") + '.png'
            img_temp = save_img(img_name, save_img_name, box_str)
            if img_temp:
                write_latex += '\n'
                save_math.write(write_latex)
                save_math.flush()
                cnt += 1

    print(cnt)


def main(save_math_txt, pix2tex_token_path, train_save_path, val_save_path):
    start_path = 'C:\\Users\\user\\Downloads\\AIHUB_Latex_data\\Data\\'
    save_path = '../../AIHUB_temp/train/'
    save_math = open(save_math_txt, 'w', encoding='utf-8')
    pre_token = open(pix2tex_token_path, 'r', encoding='utf-8')
    json_data = json.load(pre_token)

    vocab = json_data['model']['vocab']
    print(vocab)
    cnt = 0
    for txt in os.listdir('./final_train/'):
        t_v = os.path.splitext(txt)[0].split('_')[2]
        if t_v == 'tl':
            save_path = train_save_path
        else:
            save_path = val_save_path

        new_token = open('./final_train/' + txt, 'r', encoding='utf-8').readlines()

        for line in new_token:
            write_latex = line.split('\t')[1]#.replace('\\', '')
            latex = write_latex.split('\n')[0]
            latex_to_token = latex.split()
            temp = True
            for token in latex_to_token:
                if token not in vocab:
                    temp = False
                    break
            if temp:
                img_name = line.split('\t')[0]
                # load_img = img_name
                box_str = line.split('\t')[2]
                save_img_name = save_path + '/' + str(cnt).rjust(7, "0") + '.png'
                img_temp = save_img(img_name, save_img_name, box_str)
                if img_temp:
                    save_math.write(write_latex)
                    save_math.flush()
                    cnt += 1

    print(cnt)


def test():
    start_path = 'C:\\Users\\user\\Downloads\\AIHUB_Latex_data\\Data\\Training\\image\\'
    string1 = "M_HS_h1_1010_00069.png\ta x - y + 6 = 0\t[[0, 0], [147, 0], [147, 50], [0, 50]]"
    img_name = 'M_HS_h1_1010_00069.png'
    # img_name_split = img_name.split('_')
    # load_img = start_path
    # load_img += 'HS'
    # load_img += '_'
    # load_img += img_name_split[2]
    # load_img += '\\'
    # load_img += img_name_split[2]
    # load_img += '\\'
    # load_img += img_name_split[3]
    # load_img += '\\'
    # load_img += img_name
    load_img = 'C:\\Users\\user\\Downloads\\AIHUB_Latex_data\\Data\\Training\\image\\HS_h1\\h1\\1010\\M_HS_h1_1010_00069.png'
    write_latex = string1.split('\t')[1].replace('\\', '')
    latex = write_latex.split('\n')[0]

    box = string1.split('\t')[2]
    # string2 = "M_HA_h1_1010_00001.png\ta x - y + 6 = 0\t[[152, 201], [152, 256], [376, 255], [376, 200]]"
    img_temp = save_img(load_img, './test_img.png', box)
    if img_temp:
        print('true')
    #     save_math.write(write_latex)
    #     save_math.flush()
    #     cnt += 1


if __name__ == '__main__':
    # main()

    test()
