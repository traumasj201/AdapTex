import json

import sys

sys.setrecursionlimit(10 ** 6)
import os
import re


def search(filepath, cnt, total_log, n_cnt):
    for file in os.listdir(filepath):
        full_file_path = os.path.join(filepath, file)
        if os.path.isdir(full_file_path):
            search(full_file_path, cnt, total_log, n_cnt)
        else:
            if os.path.splitext(full_file_path)[1] == '.json':
                json_file = open(full_file_path, 'r', encoding='utf-8')
                json_data = json.load(json_file)
                print(full_file_path)
                for data in json_data:
                    formul = data['latex']
                    if (formul is not None) and \
                            not re.search("[가-힣ㄱ-ㅎ]", formul) and \
                            (not re.search("[①-⑳㉠-㉻ⓐ-ⓩ]", formul)):
                        if any(s in formul for s in ['\n']):
                            print(full_file_path, json_data)
                            n_cnt[0] += 1
                            continue
                    cnt[0] += 1
                    write_str = formul.strip("$")
                    write_str = write_str.replace('○', '\\circ')
                    write_str = write_str.replace('△', '\\triangle')
                    write_str = write_str.replace('▲', '\\triangle')
                    write_str = write_str.replace('□', '\\square')
                    write_str = write_str.replace('☆', '\\star')
                    write_str = write_str.replace('◈', '\\diamond')
                    write_str = write_str.replace('◇', '\\diamond')
                    write_str = write_str.replace('、', ',')
                    write_str = write_str.replace('\r', '\ r')
                    write_str = write_str.replace('\t', '\ t')
                    write_str = write_str.replace('\b', '\ b')
                    write_str = write_str.replace('\$', '$')
                    write_str = write_str.replace('×', 'x')
                    write_str = write_str.replace('°', '^{\\circ}')
                    write_str = write_str.replace('^\\circ', '^{\\circ}')
                    write_str = write_str.replace('\\left (', '\\left(')
                    write_str = write_str.replace('\\right )', '\\right)')
                    write_str = write_str.replace('◎', '\\circledcirc')
                    write_str = re.sub(r'\\(left|right)?\\(lvert|rvert|vert)', r'\\vert', write_str)
                    write_str = write_str.replace('\\lvert', '\\vert')
                    write_str = write_str.replace('\\rvert', '\\vert')
                    write_str = write_str.replace('\\dfrac', '\\frac')
                    write_str = write_str.replace('\\cfrac', '\\frac')
                    write_str = write_str.replace('\\Beta', '\\beta')
                    write_str = write_str.replace('\\cosec', '\\csc')
                    write_str = write_str.replace('\\roman', '\\romannumeral')
                    write_str = write_str.replace('\\operatorname*', '\\operatorname')
                    write_str = re.sub(r'\\(left|right)?\\(lVert|rVert|Vert)', r'\\Vert', write_str)
                    write_str = write_str.replace('\\lVert', '\\Vert')
                    write_str = write_str.replace('\\rVert', '\\Vert')
                    write_str = write_str.replace('\\omicron', 'o')

                    if len(write_str) > 0:
                        # print(cnt, json_data)

                        full_file_path2 = os.path.split(full_file_path)[0].replace('JSON', 'background_images')
                        full_file_path2 += '\\'
                        full_file_path2 += data['filename']
                        total_log.write(full_file_path2)
                        total_log.write('\t')
                        total_log.write(write_str)
                        total_log.flush()
                        total_log.write('\n')
                        total_log.flush()


if __name__ == '__main__':
    start_path = 'C:\\Users\\user\\Downloads\\AIDA_data\\'
    cnt = [0]
    n_cnt = [0]

    log_file_1 = 'all_data_aida.txt'
    total_log = open(log_file_1, 'w', encoding='utf-8')
    search(start_path, cnt, total_log, n_cnt)
    print(cnt[0])
    print(n_cnt[0])
    from tokenizer_x import token_x

    log_file_2 = 'mmodify_p.txt'
    token_x(log_file_1, log_file_2, is_aihub=False)
    from oxcheck import oxcheck

    log_file_3 = './final_text.txt'
    oxcheck(log_file_1, log_file_2, log_file_3)
    from token_compare import gen_img_txt_aida

    save_path = '../../AIDA_temp/img/'
    save_math_txt = '../../AIDA_temp/math.txt'

    gen_img_txt_aida(save_math_txt, log_file_3, save_path)

    import os
    import random

    total_folder = save_path
    train_folder = 'C:\\Users\\user\\Downloads\\LaTeX-OCR-main\\AIDA\\train\\'
    val_folder = 'C:\\Users\\user\\Downloads\\LaTeX-OCR-main\\AIDA\\val\\'
    test_folder = 'C:\\Users\\user\\Downloads\\LaTeX-OCR-main\\AIDA\\test\\'

    total_list = [f for f in os.listdir(total_folder) if f.endswith('.png')]
    print(len(total_list))
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
        source_path = os.path.join(total_folder, img)
        target_path = os.path.join(train_folder, img)
        os.rename(source_path, target_path)
    for img in val_list:
        source_path = os.path.join(total_folder, img)
        target_path = os.path.join(val_folder, img)
        os.rename(source_path, target_path)
    for img in test_list:
        source_path = os.path.join(total_folder, img)
        target_path = os.path.join(test_folder, img)
        os.rename(source_path, target_path)