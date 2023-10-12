def oxcheck(log_file_1, log_file_2, log_file_3):
    file1 = open(log_file_2, 'r', encoding='utf-8').readlines()
    file2 = open(log_file_1, 'r', encoding='utf-8').readlines()

    print(len(file1), len(file2))
    final_text = open(log_file_3, 'w', encoding='utf-8')
    cnt = 0
    for i, l in enumerate(file1):
        file1_str = l.replace(' ', '')
        file2_str = file2[i].replace(' ', '')
        if file1_str != file2_str:
            cnt += 1
            print('oxchek-----------------------------------------------------------------')
            print(l.split('\t')[1])
            # print(file2[i].split('\t')[1][:-1])
            print(file2[i])
            # print('-----------------------------------------------------------------')
            #
            # for i, s1 in enumerate(file1_str):
            #     print(file1_str[i])
            # for i, s2 in enumerate(file2_str):
            #     print(file2_str[i])
            # print(len(file1_str), len(file2_str))
            # print(file1_str[len(file1_str)-1])
            print('==========================================================')
        else:
            final_text.write(l)

    print(cnt)
