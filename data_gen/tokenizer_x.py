def token_x(log_file_1 , log_file_2, is_aihub=True):
    logfile = open(log_file_1, 'r', encoding='utf-8').readlines()
    savefile = open(log_file_2, 'w', encoding='utf-8')

    tokenlist = [
        ' ', 't', 'O', '', '0', '6', ';', 'g', 'I', 'f', '<', '/', 'X', 'b', 'K', 'V', '!', 'Z',
        'M', '?', 'v', 'P', 'n', 'Q', "'", '5', 'N', '|', '4', 'j', 'L', 'A', 'w', ',',
        'D', '&', 'y', '$', '+', '3', 'G', '>', 't', '1', '*', 'i', '"', 'z', 'Y', '8',
        'u', '9', 'q', 'J', 'm', 'R', 'p', 'C', '7', 'r', ':', 'H', 'k', 'h', '2', 'F',
        's', '=', 'T', '.', 'S', 'W', '%', 'x', 'a', 'U', 'o', '}', '-', 'd', '{', 'c',
        'E', '^', 'B', 'l', 'e', '(', ')', '_', '~', '[', ']', '\\{', '\\}', '\\,', '@',
        '\\prod', '\\downarrow', '\\widehat',
        '\\iiint', '\\ddot', '\\supsetneq',
        '\\log', '\\dot', '\\sqrt', '\\ominus', '\\leqq', '\\tanh', '\\subset',
        '\\pm', '\\oint', '\\rightharpoonup', '\\right|', '\\tau', '\\underset',
        '\\not', '\\cos', '\\theta', '\\rho', '\\epsilon', '\\roman1', '\\ln',
        '\\cong', '\\sec', '\\zeta', '\\rightleftarrows', '\\wedge', '\\prime',
        '\\left|', '\\nabla', '\\stackrel', '\\leq', '\\cdots', '\\cup', '\\square',
        '\\kappa', '\\exists', '\\gamma', '\\in', '\\eta', '\\right.', '\\vdots',
        '\\left\\|', '\\uparrow', '\\neq', '\\phi', '\\Rightarrow',
        '\\\\', '\\frac', '\\because', '\\sim', '\\perp', '\\nsubseteq', '\\Psi',
        '\\cosec', '\\fallingdotseq', '\\Pi', '\\sinh', '\\beta', '\\pi', '\\nu',
        '\\subsetneq', '\\forall', '\\tan', '\\notin', '\\xi', '\\upsilon',
        '\\rightarrow', '\\angle', '\\backslash', '\\sum', '\\nexists', '\\frown',
        '\\div', '\\cot', '\\right|', '\\overrightarrow', '\\Gamma', '\\doteq',
        '\\varnothing', '\\mp', '\\right)', '\\delta', '\\leftarrow',
        '\\overleftrightarrow', '\\searrow', '\\right\\}', '\\equiv', '\\geq',
        '\\propto', '\\vee', '\\sigma', '\\cap', '\\supset', '\\Theta', '\\sin',
        '\\partial', '\\csc', '\\to', '\\hbar', '\\lim', '\\cdot', '\\left[',
        '\\left.', '\\Lambda', '\\left\\{', '\\mu', '\\roman2',
        '\\alpha', '\\geqq', '\\int', '\\right]', '\\therefore', '\\subseteq',
        '\\leftrightarrow', '\\overline', '\\odot', '\\varphi', '\\times',
        '\\min', '\\iint', '\\oplus', '\\widetilde', '\\simeq', '\\ni', '\\omicron',
        '\\Delta', '\\Leftarrow', '\\lambda', '\\varrho', '\\Leftrightarrow', '\\roman4',
        '\\Phi', '\\psi', '\\omega', '\\coth', '\\Omega', '\\roman5', '\\left(',
        '\\triangle', '\\circ', '\\infty', '\\otimes', '\\approx', '\\max',
        '\\cosh', '\\quad', '\\limits', '\\end{array}', '\\leqslant',
        '\\text', '\\vec', '\\mid', '\\operatorname', '\\ell', '\\bar', '\\;', '\\p',
        '\\begin{array}', '\\end{matrix}', '\\begin{matrix}',
        '\\backsim', '\\mathrm', '\\ldots', '\\&', '\\leadsto', '\\qquad', '\\gt', '\\le',
        '\\mathcal', '\\rm', '\\,',
        '\\ge', '\\bigtriangleup', '\\lceil', '\\dashrightarrow',
        '\\longrightarrow', '\\nearrow', '\\', '\\lt',  '\\left\\lceil', '\\right\\rceil','\\rceil'
        '\\Box', '\\bigcirc', '\\star', '\\%', '\\bigstar', '\\langle', '\\rangle',
        '\\underline', '\\hline', '\\overset', '\\degree', '\\bullet', '\\large', '\\LARGE', '\\|', '\\displaystyle',
        '\\fbox', '\\phantom', '\\cancel', '\\mathbb', '\\ne',
        '\\left\\vert', '\\right\\vert', '\\diamond', '\\fcolorbox', '\\mathsf', '\\smash', '\\ast',
        '\\mathbf', '\\boxed', '\\vert', '\\hat', '\\begin{aligned}', '\\end{aligned}', '\\longleftarrow', '\\lvert',
        '\\rvert',  '\\emptyset', '\\sub', '\\space', '\\chi', '\\hookrightarrow', '\\Beta', '\\_', '\\boldsymbol',
        '\\curvearrowright', '\\Sigma', '\\end{cases}', '\\begin{cases}', '\\left\\langle', '\\rightsquigarrow',
        '\\diagup', '\\parallel', '\\right\\rangle', '\\longleftrightarrow', '\\!',
        #'\\rceil_{0}',
    ]
    for line in logfile:
        new_line = []
        word = []
        formul = line.split('\t')[1]
        file_name = line.split('\t')[0]
        if is_aihub:
            box = line.split('\t')[2]
        for i, s in enumerate(formul):
            word.append(s)
            # print(str(word))
            new_word = ''.join(word)
            # print(str(new_word))
            if new_word == '\\':
                if formul[i:i + 2] != '\\ ':
                    continue

            if new_word == '\\in':
                if formul[i - 1:i + 3] == 'inft' or formul[i - 1:i + 2] == 'int':
                    continue
            if new_word == '\\lim':
                if formul[i - 2:i + 2] == 'limi':
                    continue
            if new_word == '\\le':
                if formul[i - 1:i + 2] == 'lef':
                    continue
                elif formul[i - 1:i + 2] == 'leq':
                    continue
                elif formul[i - 1:i + 2] == 'lea':
                    continue
            if new_word == '\\leq':
                if formul[i - 2:i + 2] == 'leqq':
                    continue
                elif formul[i - 2:i + 2] == 'leqs':
                    continue
            if new_word == '\\tan':
                if formul[i - 2:i + 2] == 'tanh':
                    continue
            if new_word == '\\p':
                p_arr = ['pr', 'ph', 'pa', 'pe', 'pi', 'ps', 'pm']
                if formul[i:i + 2] in p_arr:
                    continue
            if new_word == '\\ne':
                if formul[i - 1:i + 2] == 'neq':
                    continue
                elif formul[i - 1:i + 2] == 'nex':
                    continue
                elif formul[i - 1:i + 2] == 'nea':
                    continue
            if new_word == '\\sub':
                if formul[i - 2:i + 2] == 'subs':
                    continue
            if new_word == '\\subset':
                if formul[i - 5:i + 2] == 'subsetn':
                    continue
                elif formul[i - 5:i + 2] == 'subsete':
                    continue

            if str(new_word) in tokenlist:
                if len(new_line) > 0 and new_word != ' ':
                    # print(new_line)
                    new_line.append(' ')
                new_line.append(new_word)
                word = []

        # print(new_line)
        if len(new_line) > 0:
            # print('new',new_line)
            # print('new', ''.join(new_line))
            savefile.write(file_name)
            savefile.write('\t')
            savefile.write(''.join(new_line).replace('  ', ' '))
            if is_aihub:
                savefile.write('\t')
                savefile.write(box)
            else:
                savefile.write('\n')
            savefile.flush()
        else:
            savefile.write(file_name)
            savefile.write('\t')

            if is_aihub:
                savefile.write(' ')
                savefile.write('\t')
                savefile.write(box)
            else:
                savefile.write('\n')
            savefile.flush()
            print(new_line)
            print(line)
        # print(formul)
