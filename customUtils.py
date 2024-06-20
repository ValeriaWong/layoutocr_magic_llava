import requests

def fetch_image_info(file_path):
    """异步获取图像信息的API调用"""

    url = 'http://v.onlyax.com:38200/upload_file_and_detect_text'
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)

        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()
            raise Exception("API调用失败")
