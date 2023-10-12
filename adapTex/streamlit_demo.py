import time

import streamlit as st
from streamlit_core import Pix2TexModel
from streamlit_cropper import st_cropper
from PIL import Image, ExifTags
from urllib.parse import quote
import matplotlib.pyplot as plt
import matplotlib as mpl
import io
import pyperclip


## 기능 함수 ##
# LaTeX문법 이미지로 변환
def latex_to_image(latex_str):
    # 일단 대략적인 크기로 그림과 축을 생성
    fig, ax = plt.subplots(figsize=(12, 3))

    # LaTeX 문자열로 텍스트를 생성
    txt = ax.text(0.5, 0.5, f'${latex_str}$', size=15, va='center', ha='center')

    # 텍스트의 바운딩 박스의 너비와 높이를 얻음
    fig.canvas.draw()  # 이를 호출해야 get_window_extent()가 정확한 값을 반환
    bbox = txt.get_window_extent(fig.canvas.get_renderer())
    width, height = bbox.width, bbox.height
    width /= fig.dpi  # 인치 단위로 변환
    height /= fig.dpi  # 인치 단위로 변환

    # 얻은 너비와 높이로 그림의 크기를 재조정
    fig.set_size_inches(width + 1, height + 1)  # 여백을 위해 약간 추가
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.2)
    buf.seek(0)

    plt.close(fig)
    return buf


def clear_state():
    if "predict_latex" in st.session_state:
        del st.session_state.predict_latex


if __name__ == '__main__':
    try:

        st.title("수식 변환기📸")
        # 모델 초기화

        model = Pix2TexModel()

        # matplotlib의 폰트 설정 변경

        mpl.rcParams['text.usetex'] = False
        ## 파일 업로드 작업 ##
        # 사용자로부터 이미지 입력
        uploaded_file = st.file_uploader("", type=["png", "jpg"], key='uploaded_file', on_change=clear_state)

        if st.session_state.uploaded_file is not None:
            img = Image.open(uploaded_file)  # 이미지 열기
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = dict(img._getexif().items())
                
                if exif[orientation] == 3:
                    img = img.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    img = img.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    img = img.rotate(90, expand=True)
            except (AttributeError, KeyError, IndexError):
                # cases: image don't have getexif
                pass
            crop_col, button_col = st.columns([10, 1])
            # 이미지 크롭
            cropped_img = st_cropper(img_file=img, realtime_update=True, box_color="green")

            # 전체 이미지 사용 토글
            use_full = st.toggle("전체 이미지 사용")
            if use_full:
                # 전체 이미지
                final_img = uploaded_file
            else:
                # 이미지 자르기
                final_img = cropped_img

            image_container = st.container()
            caption = "최종 입력 이미지"

            # 캡션을 가운데 정렬하는 HTML 및 CSS 스타일 사용
            centered_text = f'<div style="display: flex; justify-content: center;"><p style="font-size:18px;">{caption}</p></div>'
            image_container.markdown(centered_text, unsafe_allow_html=True)
            image_container.image(final_img, use_column_width=True)


            ## 예측 부분 ##
            if st.button("수식 변환", key="Start_btn"):

                if "predict_latex" in st.session_state:
                    del st.session_state.predict_latex
                    del st.session_state.latex_input_text
                with st.spinner("분석중....."):


                    prediction = model.predict(final_img, use_full)

                    st.session_state.predict_latex = prediction
                    # data = st.text_input("수식 수정:",st.session_state.predict_latex)

            # 수식이 세션에 저장되어있다면 표시
            if "predict_latex" in st.session_state:

                if 'latex_input_text' in st.session_state:
                    latex_input_text_str = st.session_state.latex_input_text
                else:
                    latex_input_text_str = st.session_state.predict_latex

                st.latex(latex_input_text_str)
                text_col, copy_col = st.columns([9.7, 1])
                st.session_state.predict_latex = text_col.text_input("수식 수정:", latex_input_text_str, key='latex_input_text',
                                                                 label_visibility="collapsed")
                print('수정중_', st.session_state.latex_input_text)

                if copy_col.button("복사", key='clipboard_btn'):
                    # 클립보드에 텍스트 복사
                    pyperclip.copy(st.session_state.predict_latex)
                    toast_msg = st.toast("수식 복사 완료!", icon="✂")
                
                with st.expander("내보내기"):
                    # 울프람알파 내보내기
                    encoded_prediction = quote(st.session_state.predict_latex)  # URL 또는 다른 web에 보내기위한 인코딩
                    wolfram_url = f"https://www.wolframalpha.com/input/?i={encoded_prediction}"
                    button_code = f"""
                    <a href="{wolfram_url}" target="_blank" style="display: inline-block; text-decoration: none; background-color: #F96932; color: white; padding: 8px 16px; border-radius: 4px;">WolframAlpha</a>
                    """
                    st.markdown(button_code, unsafe_allow_html=True)


    except KeyboardInterrupt:
        print('Ctrl + C 중지 메시지 출력')
