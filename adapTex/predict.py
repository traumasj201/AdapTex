import argparse
import torch
import yaml
import cv2
from munch import Munch
from models import get_model
from utils import *
from dataset.transforms import test_transform
from transformers import PreTrainedTokenizerFast


def detokenize(tokens, tokenizer):
    toks = [tokenizer.convert_ids_to_tokens(tok) for tok in tokens]
    for b in range(len(toks)):
        for i in reversed(range(len(toks[b]))):
            if toks[b][i] is None:
                toks[b][i] = ''
            toks[b][i] = toks[b][i].replace('Ġ', ' ').strip()
            if toks[b][i] in (['[BOS]', '[EOS]', '[PAD]']):
                del toks[b][i]
    return toks

def preprocessing(_img):
    max_h = 192
    max_w = 672


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
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def img2tensor(_img):
    return test_transform(image=cv2.cvtColor(_img, cv2.COLOR_BGR2RGB))['image'][:1].unsqueeze(0)


def post_process(s: str):
    """Remove unnecessary whitespace from LaTeX code.

    Args:
        s (str): Input string

    Returns:
        str: Processed image
    """
    text_reg = r'(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})'
    letter = '[a-zA-Z]'
    noletter = '[\W_^\d]'
    names = [x[0].replace(' ', '') for x in re.findall(text_reg, s)]
    s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
    news = s
    while True:
        s = news
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, noletter), r'\1\2', s)
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, letter), r'\1\2', news)
        news = re.sub(r'(%s)\s+?(%s)' % (letter, noletter), r'\1\2', news)
        if news == s:
            break
    return s



@torch.no_grad()
def prediction(model, img, args):
    """evaluates the model. Returns bleu score on the dataset

    Args:
        model (torch.nn.Module): the model
        img : cv2 img
        args : config yaml
    Returns:
        str : latex
    """

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
    dec = model.generate(img2tensor(img).to(args.device), temperature=args.temperature)
    # pred = detokenize(dec, tokenizer) # 토큰 list 형식 으로 출력
    # pred = token2str(dec, tokenizer)[0] # 입력 형식 대로 출력(토큰 사이 띄어쓰기 존재)
    pred = post_process(token2str(dec, tokenizer)[0]) # 최종 Latex
    return pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('--config', default='./model/setting/config.yaml', help='path to yaml config file', type=str)
    parser.add_argument('-c', '--checkpoint', default='./model.pth', type=str, help='path to model checkpoint')
    parser.add_argument('-t', '--temperature', type=float, default=.333, help='sampling emperature')
    parser.add_argument('-i', '--image', type=str, default='./test2img.png', help='sampling emperature')
    # parser.add_argument('-n', '--tokenizer', type=str, default='./adapTex/model/dataset/tokenizer.json', help='sampling emperature')
    parsed_args = parser.parse_args()

    with open(parsed_args.config, 'r', encoding='utf-8') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    args = parse_args(Munch(params))
    args.temperature = parsed_args.temperature
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_img = cv2.imread(parsed_args.image)
    input_img = preprocessing(input_img)

    model = get_model(args)
    model.load_state_dict(torch.load(parsed_args.checkpoint, args.device))


    model.eval()


    pred = prediction(model, input_img, args)
    print(pred)


