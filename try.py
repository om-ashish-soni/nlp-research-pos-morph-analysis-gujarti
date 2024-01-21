import tempfile
import os
from huggingface_hub import hf_hub_url, hf_hub_download
from dotenv import load_dotenv


def get_dir_path():
    temp_dir = tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

def download_file():
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    repo_file_name="HfApiUploaded_GUJ_SPLIT_POS_MORPH_ANAYLISIS-v6.0-model.pth"
    temp_dir=get_dir_path()
    model_filepath=os.path.join(temp_dir,repo_file_name)
    hf_hub_download(
        repo_id="om-ashish-soni/research-pos-morph-gujarati-6.0",
        filename=repo_file_name,
        local_dir = temp_dir,
        token=HF_TOKEN
    )
    return model_filepath

download_file()