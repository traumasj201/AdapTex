import os
import json

import sys

sys.setrecursionlimit(10 ** 6)

import re


def search(filepath, cnt, total_log, n_cnt, is_hand_w, is_grade):
    for file in os.listdir(filepath):
        full_file_path = os.path.join(filepath, file)
        # print(full_file_path)
        if os.path.isdir(full_file_path):
            if len(full_file_path.split('\\')) == 6:
                split_txt = full_file_path.split('\\')[5]
                if is_hand_w:
                    if split_txt.split('_')[0] == 'h':
                        if split_txt.split('_')[1] in is_grade:
                            search(full_file_path, cnt, total_log, n_cnt, is_hand_w, is_grade)
                else:
                    if split_txt.split('_')[0] != 'h':
                        if split_txt.split('_')[0] in is_grade:
                            search(full_file_path, cnt, total_log, n_cnt, is_hand_w, is_grade)
            else:
                search(full_file_path, cnt, total_log, n_cnt, is_hand_w, is_grade)
        else:
            if os.path.splitext(full_file_path)[1] == '.json':

                json_file = open(full_file_path, 'r', encoding='utf-8')
                json_data = json.load(json_file)
                # print(json_data)
                # print(json_data['segments'])

                segments = json_data['segments']
                # if len(segments) == 1:
                for segment in segments:
                    if segment['type_detail'] == '수식':
                        formul = segment['equation']
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
                                full_file_path = full_file_path.replace('label', 'image')
                                total_log.write(full_file_path.replace('.json', '.png'))
                                total_log.write('\t')
                                total_log.write(write_str)
                                total_log.flush()
                                total_log.write('\t')
                                total_log.write(str(segment['box']))
                                total_log.write('\n')
                                total_log.flush()
                        # else:
                        #     print(cnt, json_data)

            # sys.exit()


def file_move(filepath):
    for file in os.listdir(filepath):
        full_file_path = os.path.join(filepath, file)

        if os.path.isdir(full_file_path):
            file_move(full_file_path)
        else:
            if os.path.splitext(full_file_path)[1] == '.json' or os.path.splitext(full_file_path)[1] == '.png':

                move_folder = full_file_path.replace('손고1', 'h_high1')
                move_folder = move_folder.replace('손고2', 'h_high2')
                move_folder = move_folder.replace('손고3', 'h_high3')
                move_folder = move_folder.replace('고1', 'high1')
                move_folder = move_folder.replace('고2', 'high2')
                move_folder = move_folder.replace('고3', 'high3')

                move_folder = move_folder.replace('손중1', 'h_mid1')
                move_folder = move_folder.replace('손중2', 'h_mid2')
                move_folder = move_folder.replace('손중3', 'h_mid3')

                move_folder = move_folder.replace('중1', 'mid1')
                move_folder = move_folder.replace('중2', 'mid2')
                move_folder = move_folder.replace('중3', 'mid3')

                move_folder = move_folder.replace('손초4', 'h_ele4')
                move_folder = move_folder.replace('손초5', 'h_ele5')
                move_folder = move_folder.replace('손초6', 'h_ele6')
                move_folder = move_folder.replace('초4', 'ele4')
                move_folder = move_folder.replace('초5', 'ele5')
                move_folder = move_folder.replace('초6', 'ele6')

                move_folder = move_folder.replace('TL_', '')
                move_folder = move_folder.replace('TS_', '')
                move_folder = move_folder.replace('VL_', '')
                move_folder = move_folder.replace('VS_', '')
                os.makedirs(os.path.split(move_folder)[0], exist_ok=True)
                os.rename(full_file_path, move_folder)
            else:
                print(full_file_path)


if __name__ == '__main__':
    is_hand_w = False
    is_grade = ['high1', 'high2', 'high3', 'mid1', 'mid2', 'mid3', 'ele4', 'ele5', 'ele6']
    start_path = 'C:\\aihub_data\\Data\\'

    # file_move(start_path)

    cnt = [0]
    n_cnt = [0]
    is_train = True  # True : train, False : val
    log_file_1 = 'all_data_aida.txt'
    total_log = open('all_data_p.txt', 'w', encoding='utf-8')
    search(start_path, cnt, total_log, n_cnt, is_hand_w, is_grade)

    print(cnt[0])
    print(n_cnt[0])

    from tokenizer_x import token_x

    log_file_2 = 'mmodify_p.txt'
    token_x(log_file_1, log_file_2)
    from oxcheck import oxcheck

    log_file_3 = './final_text.txt'
    oxcheck(log_file_1, log_file_2, log_file_3)
    #
    from token_compare import gen_img_txt

    save_path = '../../AIHUB_temp/train/'
    save_math_txt = '../../AIHUB_temp/math.txt'
    pix2tex_token_path = '../adapTex/model/dataset/tokenizer.json'
    gen_img_txt(save_math_txt, pix2tex_token_path, log_file_3, save_path)
    import os
    import random

    total_folder = 'C:\\Users\\user\\Downloads\\LaTeX-OCR-main\\AIHUB_temp\\train\\'
    train_folder = 'C:\\Users\\user\\Downloads\\LaTeX-OCR-main\\AIHUB_p\\train\\'
    val_folder = 'C:\\Users\\user\\Downloads\\LaTeX-OCR-main\\AIHUB_p\\val\\'
    test_folder = 'C:\\Users\\user\\Downloads\\LaTeX-OCR-main\\AIHUB_p\\test\\'

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
