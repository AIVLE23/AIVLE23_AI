import re

def compare(origin,transcription):

    nSen1 = origin.split()
    
    nSen2 = re.sub('[^가-힣\s]', '', transcription)
    nSen2 = re.sub('\s+', ' ', nSen2)
    nSen2 = transcription.split()

    diff = []
    for i in range(0,len(nSen1)):
        if nSen1[i]!=nSen2[i]:
            diff.append([nSen2[i],nSen1[i]])

    # #틀린 부분 출력
    # print(diff)
