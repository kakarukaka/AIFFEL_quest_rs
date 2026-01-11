import os

# API Keys
GOOGLE_API_KEY = ""
PUBLIC_DATA_KEY_DECODED = ''

# Paths
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRUG_CSV_FILE = os.path.join(DATA_DIR, "drug.csv")
DRUG_NAME_CSV = os.path.join(DATA_DIR, "drug_name.csv")
FAISS_INDEX_DIR = os.path.join(DATA_DIR, "silver_faiss")
DRUG_JSON_CREDENTIALS = os.path.join(DATA_DIR, "drug.json")

# URL
OPENAPI_URL = 'http://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList'
