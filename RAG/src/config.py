import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Default to the RAG folder root so the project runs on any machine.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    BASE_DIR = os.getenv("MEDCHAT_RAG_BASE_DIR", str(PROJECT_ROOT))
    DATA_DIR = f"{BASE_DIR}/data"
    CHUNK_FILES = [
        f"{DATA_DIR}/BoYTe200_v3.json",
        f"{DATA_DIR}/NHIKHOA2.json",
        f"{DATA_DIR}/PHACDODIEUTRI_2016.json"
    ]    
    EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder" # Model embedding - Nhẹ và nhanh
    # EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
    # EMBEDDING_MODEL = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
    
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
    
    LLM_MODEL = "gemini-2.5-flash" 
    K_RETRIEVE = 3 # Số Document muốn truy
    TEMPERATURE = 0 
