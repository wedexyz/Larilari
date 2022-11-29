import cv2
from datetime import datetime

def get_Center(x, y, w, h): 
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


#thres = 0.45 # Threshold to detect object
thres = 0.57

cap = cv2.VideoCapture('lari.mp4')



classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


while True:
    success,img = cap.read()
    ims= img.copy()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    #print(classIds,bbox)
    #cv2.line(img,(1280,580),(0,580),(0,255,0),2)
    cv2.line(img,(340,0),(340,720),(0,255,0),2)


    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            if classNames[classId-1] =='person' :
                ## Menghitung Jarak ##
                x = int(box[0])
                y = int(box[1])
                w = int(box[2])
                h = int(box[3])

                cx=int((x+x+w)/2)
                cy=int((y+y+h)/2)
                mid_point = cx,cy
                #center = int((x + w)/2), int((h + y)/2)
                print(mid_point)
            
            
                lebar = w / 20
                yeye = y + h/3
                reye = x + (w/2) - (w/5)
                leye = x + (w/2) + (w/5)
                space = leye - reye
                f = 690
                r = 10
                distance = f * r / space
                distance_in_cm = int(distance)
                cv2.putText(img, str(distance_in_cm)+' cm', (box[0]+200,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),1)
                cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,0),1)
                cv2.putText(img,str(round(confidence*100,2))+' %',(box[0]+100,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,255),1)
                cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
        
                if cx > 1100 :
                    cv2.imshow('jpg', ims)
                    jam = datetime. now(). strftime("%I")
                    menit = datetime. now(). strftime("%M")
                    detik = datetime. now(). strftime("%S")
                    

                    
                    cv2.imwrite(f"data/jam_{jam}menit_{menit}detik_{detik}.jpg",ims)
                    print('a')
            
    cv2.imshow("Output",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()