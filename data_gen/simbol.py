import os
import cv2


def preprocessing(_img):
    max_h = 192
    max_w = 672
    if _img.ndim == 2:
        h, w = _img.shape
    else:
        h, w, _ = _img.shape

    if w > max_w:
        ratio = max_w / w
        new_h = int(h * ratio)
        h, w = new_h, max_w

    if h > max_h:
        ratio = max_h / h
        new_w = int(w * ratio)
        h, w = max_h, new_w

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
    img = cv2.resize(_img, (new_w, new_h))

    if img.ndim == 3 and img.shape[2] == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img

    return gray_img


def file_move(filepath, cnt, imgTempPath, math_txt):
    for file in os.listdir(filepath):
        full_file_path = os.path.join(filepath, file)
        if os.path.isdir(full_file_path):
            file_move(full_file_path, cnt, imgTempPath, math_txt)
        else:
            try:
                exp = os.path.splitext(full_file_path)[1]
                if exp == '.jpg' or exp == '.png' or exp == '.bmp':  # 여기 이미지 확장자 추가

                    label = full_file_path.split('/')[-2]
                    if label == '(':
                        label = '\\left('
                    elif label == ')':
                        label = '\\right)'
                    elif label == '{':
                        label = '\\left{'
                    elif label == ')':
                        label = '\\right}'
                    elif label == '[':
                        label = '\\left['
                    elif label == ']':
                        label = '\\right]'
                    elif label in ["3", "A", "d", "N", "v", "4", "G", "q", "w", "5",
                                   "o", "R", "T", "X", "+", "6", "b", "e", "j", "p", "y", ",",
                                   "0", "7", "k", "S", "z", "-", "1", "8", "C", "f", "H", "l", "M", "=", "2", "9", "i",
                                   "u"]:
                        label = label
                    else:
                        label = '\\' + label
                    img = cv2.imread(full_file_path)
                    w, h, c = img.shape
                    if w < 11 and h < 11:
                        continue
                    gray_img = preprocessing(img)  # 프리프로세싱 호출

                    image_path = imgTempPath + str(cnt[0]).rjust(7, "0") + '.png'
                    cv2.imwrite(image_path, gray_img)

                    cnt[0] += 1
                    math_txt.write(label)
                    math_txt.write('\n')
                    math_txt.flush()

                else:
                    print(exp)
            except:
                continue


if __name__ == '__main__':
    rootpath = './extracted_images/'
    cnt = [0]
    mathPath = './symbol_math.txt'
    math_txt = open(mathPath, 'w')
    imgTempPath = './symbol_set/'
    if not os.path.isdir(imgTempPath):
        os.mkdir(imgTempPath)
    file_move(rootpath, cnt, imgTempPath, math_txt)

    train_folder = './symbol_set_train/'
    if not os.path.isdir(train_folder):
        os.mkdir(train_folder)
    val_folder = './symbol_set_val/'
    if not os.path.isdir(val_folder):
        os.mkdir(val_folder)
    test_folder = './symbol_set_test/'
    if not os.path.isdir(test_folder):
        os.mkdir(test_folder)

    total_list = [f for f in os.listdir(imgTempPath) if f.endswith('.png')]
    print(len(total_list))
    import random

    random.shuffle(total_list)
    random.shuffle(total_list)
    random.shuffle(total_list)

    num_train = int(0.8 * len(total_list))
    num_val = int(0.9 * len(total_list))

    train_list = total_list[:num_train]
    print(len(train_list))
    val_list = total_list[num_train:num_val]
    print(len(val_list))
    test_list = total_list[num_val:]
    print(len(test_list))

    for img in train_list:
        source_path = os.path.join(imgTempPath, img)
        target_path = os.path.join(train_folder, img)
        os.rename(source_path, target_path)
    for img in val_list:
        source_path = os.path.join(imgTempPath, img)
        target_path = os.path.join(val_folder, img)
        os.rename(source_path, target_path)
    for img in test_list:
        source_path = os.path.join(imgTempPath, img)
        target_path = os.path.join(test_folder, img)
        os.rename(source_path, target_path)
