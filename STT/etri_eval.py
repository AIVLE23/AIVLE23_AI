path = "C:\\Users\\user\\Downloads\\131.인공지능 학습을 위한 외국인 한국어 발화 음성 데이터\\01.데이터_new_20220719\\2.Validation\\원천데이터\VS_4. 중국어\\5. 한국문화II\CN50QA286_CN0476_20211014.wav"

import json
import urllib3
import json
import base64
import librosa
import numpy as np

def etri_eval(origin_text:str,audio,key:str):
    # openApiURL = "http://aiopen.etri.re.kr:8000/WiseASR/Pronunciation" # 영어
    openApiURL = "http://aiopen.etri.re.kr:8000/WiseASR/PronunciationKor" # 한국어

    accessKey = key
    audioFilePath = path # audio로 바꾸면 될 듯 한데 체크해야함
    languageCode = "korean"
    script = origin_text


    pcm = (librosa.load(path, sr=16000)[0] * 32767).astype(np.int16)
    audioContents = base64.b64encode(pcm).decode('utf8')

    requestJson = {   
        "argument": {
            "language_code": languageCode,
            "script": script,
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

    # print("[responseCode] " + str(response.status)) # 응답 코드 필요하다면 사용
    # print("[responBody]")
    # print(str(response.data,"utf-8"))
    
    result = json.loads(response.data)['return_object']['score']
    
    return result