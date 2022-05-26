import cv2
# to showan image lena
#img  =cv2.imread('lena.png')
# cv2.imshow("output",img)
# cv2.waitKey(0)
#img = cv2.imread('lena.png')

cap =cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)
cap.open(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# print(classNames)
# import the config file
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
WeightsPath = 'frozen_inference_graph.pb'
# open cv provides a function taht does all the work and we only need to give it config and weight paths
#not neccesary to understand
net = cv2.dnn_DetectionModel(WeightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
#bbox bounding box
while True:
    success,img=cap.read()        
    classIds, confs, bbox = net.detect(img, confThreshold=0.45)
    print(classIds, bbox)
    #this is not your usual for loop it three in one
    if len(classIds)!=0:
        for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,color=(0,255,0),thickness=2)
    cv2.imshow("output", img)
    cv2.waitKey(1)
