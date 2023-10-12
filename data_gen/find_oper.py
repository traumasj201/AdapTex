import re

# 텍스트 파일의 경로
file_path = 'C:\\Users\\user\\Downloads\\new_pdfmath.txt'

# file_path = 'C:\\Users\\user\\Downloads\\LaTeX-OCR-main\\AIHUB_temp\\math.txt'

# 정규 표현식 패턴
pattern = r'\\operatorname\s*\*?\s*\{([^}]*)\}'

pattern = r'\\operatorname\s*\*?\s*\{([^}]*)\}'

# 결과를 저장할 리스트
matches = []
matches_set = set()

# 텍스트 파일 열기
with open(file_path, 'r', encoding='utf-8') as file:
    # 파일 내용 읽기
    text = file.read()

    # 정규 표현식으로 매칭된 모든 문자열 찾기
    matches = re.findall(pattern, text)
    for match in matches:
        matches_set.add(match)

# 결과 출력
for match in matches_set:
    print(match)

print(len(matches_set))

ttt2 = '\\operatorname * { l i m }'
rrr2 = re.sub(r'\\operatorname\s*\*?\s*\{( l i m )\}', r'\\lim', ttt2)
print(rrr2)