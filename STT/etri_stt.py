import json
import urllib3
import json
import base64
import librosa
import numpy as np

def etri_stt(audio, key:str):
    
    openApiURL = "http://aiopen.etri.re.kr:8000/WiseASR/Recognition"
    accessKey = key
    
    # 허깅페이스와 동일하게 wav파일 경로를 주면 자동으로 리샘플링하고 바꿔주는거랑 일단 audio~paht에 경로 넣어주면 될 듯
    audioFilePath = path
    languageCode = "korean"

    pcm = (librosa.load(path, sr=16000)[0] * 32767).astype(np.int16)
    audioContents = base64.b64encode(pcm).decode('utf8')

    requestJson = {    
        "argument": {
            "language_code": languageCode,
            "audio": audioContents
        }
    }

    http = urllib3.PoolManager()
    response = http.request(
        "POST",
        openApiURL,
        headers={"Content-Type": "application/json; charset=UTF-8","Authorization": accessKey},
        body=json.dumps(requestJson)
    )

    # print("[responseCode] " + str(response.status)) api 응답 코드 필요하다면 이 코드 사용
    # print("[responBody]")
    # print(str(response.data,"utf-8"))
    
    result = json.loads(response.data)['return_object']['recognized']
    
    return result