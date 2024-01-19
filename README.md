# AdapTex
**해당 프로젝트는 아이펠 온라인 4기 팀 프로젝트 입니다.**
<br>
<br>
저희 프로젝트의 목표는 수학 공식이 적힌 이미지를 LaTeX 문법으로 변환하는 프로젝트 입니다.<br>
OCR을 거쳐 나온 Latex 문장을 블로그, overleaf, 검색 등 다양하게 이용 가능합니다.<br>
<br>
![img.png](img%2Fimg.png)<br>
<br>
![info.png](img%2Finfo.png)<br>
<br>
>#### [presentation](https://www.canva.com/design/DAFxJbscnIc/Ok4O_I3J78YH-EfAa2uxqA/edit?utm_content=DAFxJbscnIc&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | [paper](https://github.com/traumasj201/AdapTex/blob/main/paper/AdapTex_paper.pdf) |  [poster](https://github.com/traumasj201/AdapTex/blob/main/paper/AdapTex_poster.pdf) |

# Model

### Performance
| model | BLEU score | token accuracy  | edit distance|
|:----------:|----------:|--------------------:|-------------:|
|pix2tex|0.88|0.60|0.10|
|**AdapTex**|**0.93**|**0.81**|**0.05**|

### Architecture
- ##### encoder
![Architecture.png](img%2FArchitecture.png)


# Using the demo
- ###### model download
|model|is_af|link|
|-----|------|-----|
|af_h_model|True|[google drive](https://drive.google.com/file/d/1MPUeHb5M5aISqpZTPJSW6mrry0hlijWa/view?usp=drive_link)|
|p_h_model|False|[google drive](https://drive.google.com/file/d/17YWTUHvNi4MFilrKApxd0kHT4Cdq4exf/view?usp=drive_link)|

>모델 경로 설정은 config의 load_chkpt를 설정 해주세요

##### Streamlit으로 구현된 데모 입니다.

누구나 실행해볼 수 있습니다. 아래 절차를 따라주세요.

1. **requirements.txt**를 설치해줍니다.
```
pip3 install -r  requirements.txt
```

2. 데모를 실행하세요.
```
$ cd adapTex
streamlit run streamlit_demo.py
```

# Using the train
> 저희가 이용한 dataset들은 [구글Drive](https://drive.google.com/drive/folders/1tJE-n-DRMrPQ_OsbjSRIcgAx0qV4pYSN?usp=drive_link)에서 받으실 수 있습니다.
### create pkl
```
$ cd adapTex
python3 dataset/dataset.py -i [image folder] [image folder] -e [equations.txt] [equations.txt] -o [output_train.pkl]
python3 dataset/dataset.py -i [image folder] [image folder] -e [equations.txt] [equations.txt] -o [output_val.pkl]
```
### train
- model/settings/config.yaml 수정
```
$ cd adapTex
python3 train.py --config model/settings/config.yaml # use wandb
```
# References
### Model
[1] [pix2tex - LaTeX OCR](https://github.com/lukas-blecher/LaTeX-OCR)

[2] [[NeurIPS 2022] AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition](https://github.com/ShoufaChen/AdaptFormer)

### Dataset
[1] [PDF](https://zenodo.org/record/56198#.V2px0jXT6eA)

[2] [AIHUB](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=479)

[3] [CROHME](http://www.iapr-tc11.org/mediawiki/index.php/CROHME:_Competition_on_Recognition_of_Online_Handwritten_Mathematical_Expressions)

[4] [AIDA](https://www.kaggle.com/datasets/aidapearson/ocr-data)

[5] [CROHME_symbol](https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols)

