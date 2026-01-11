import requests
import xml.etree.ElementTree as ET
import pandas as pd
import math
from src.config import PUBLIC_DATA_KEY_DECODED, OPENAPI_URL, DRUG_CSV_FILE, DRUG_NAME_CSV

def fetch_and_save_drug_data():
    rows = []
    url = OPENAPI_URL
    
    # 초기 요청으로 전체 개수 파악
    params ={'serviceKey' : PUBLIC_DATA_KEY_DECODED,
             'pageNo' : '1', 'numOfRows' : '500', 'type' : 'xml', 'entpName' : '', 'itemName' : '',
             'itemSeq' : '', 'efcyQesitm' : '', 'useMethodQesitm' : '',
             'atpnWarnQesitm' : '', 'atpnQesitm' : '', 'intrcQesitm' : '',
             'seQesitm' : '', 'depositMethodQesitm' : '', 'openDe' : '',
             'updateDe' : '' }

    response = requests.get(url, params=params)
    response.encoding = 'utf-8'
    root = ET.fromstring(response.text)

    total_count = int(root.findtext('.//totalCount'))
    num_of_rows = int(params['numOfRows'])
    total_pages = math.ceil(total_count / num_of_rows)

    print(total_count, total_pages)

    for page in range(1, total_pages + 1):
        params['pageNo'] = str(page)

        response = requests.get(url, params=params)
        response.encoding = 'utf-8'
        root = ET.fromstring(response.text)

        for item in root.findall('.//item'):
            product_name = item.findtext('itemName')

            # [고도화] 제품명에 '어린이'가 포함되어 있으면 저장하지 않고 건너뜀
            if '어린이' in product_name:
                continue

            rows.append({
                '제조사': item.findtext('entpName'),
                '제품명': product_name,
                '품목번호': item.findtext('itemSeq'),
                '효능': item.findtext('efcyQesitm'),
                '복용방법': item.findtext('useMethodQesitm'),
                '주의사항': item.findtext('atpnQesitm'),
                '상호작용': item.findtext('intrcQesitm'),
                '부작용': item.findtext('seQesitm'),
                '보관방법': item.findtext('depositMethodQesitm'),
                '허가일': item.findtext('openDe'),
                '업데이트일': item.findtext('updateDe')
            })

    df = pd.DataFrame(rows)
    df.to_csv(DRUG_CSV_FILE, index=False)
    
    # drug_name.csv 생성
    df[['제품명']].to_csv(DRUG_NAME_CSV, index=False)
    
    return df

if __name__ == "__main__":
    fetch_and_save_drug_data()
