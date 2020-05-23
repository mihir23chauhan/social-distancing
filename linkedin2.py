import time
import cv2
import numpy as np
import math


confid=0.5
thresh= 0.5

vid_path="./videos/walking_people.mp4"
angle_factor=0.8
H_zoom_factor=1.2

def dist(c1,c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

def T2S(T):
    S=abs(T/((1+T**2)**0.5))
    return S

def T2C(T):
    c= abs(1/((1+T**2)**0.5))
    return c


def isclose(p1,p2):

    c_d = dist(p1[2], p2[2])
    if(p1[1]<p2[1]):
        a_w = p1[0]
        a_h = p1[1]
    else:
        a_w = p2[0]
        a_h = p2[1]

    T = 0
    try:
        T=(p2[2][1]-p1[2][1])/(p2[2][0]-p1[2][0])
    except ZeroDivisionError:
        T = 1.633123935319537e+16
    S = T2S(T)
    C = T2C(T)
    d_hor = C*c_d
    d_ver = S*c_d
    vc_calib_hor = a_w*1.3
    vc_calib_ver = a_h*0.4*angle_factor
    c_calib_hor = a_w *1.7
    c_calib_ver = a_h*0.2*angle_factor
    # print(p1[2], p2[2],(vc_calib_hor,d_hor),(vc_calib_ver,d_ver))
    if (0<d_hor<vc_calib_hor and 0<d_ver<vc_calib_ver):
        return 1
    else:
        return 0

lablesPath="./yolo-coco/coco.names"
LABLES=open(lablesPath).read().strip().split("\n")

np.random.seed(42)

weightsPath = "./yolo-coco/yolov3.weights"
configPath = "./yolo-coco/yolov3.cfg"

net=cv2.dnn.readNetFromDarknet(configPath,weightsPath)
ln=net.getLayerNames()
ln=[ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
FR=0
vs=cv2.VideoCapture(vid_path)
writer=None
(W,H)=(None,None)

f1=0
q=0
while True:

    (grabbed, frame)=vs.read()
    if not grabbed:
        break
    if W is None or H is None:
        (H,W)=frame.shape[:2]
        FW=W
        if(W<1075):
            FW = 1075
        FR = np.zeros((H+210,FW,3), np.uint8)

        col = (255,255,255)
        FH = H + 210
    FR[:]=col

    blob=cv2.dnn.blobFromImage(frame,1/255.0,(416,416),swapRB=True, crop=False)

    net.setInput(blob)
    start= time.time()
    layerOutputs=net.forward(ln)
    end=time.time()

    boxes=[]
    confidences=[]
    classIDs=[]

    for output in layerOutputs:
        for detection in output:

            scores= detection[5:]
            classID= np.argmax(scores)
            confidence=scores[classID]
            if LABLES[classID]=="person":

                if confidence> confid:
                    box=detection[0:4] * np.array([W, H, W,H])
                    (centerX, centerY, width, height)= box.astype("int")

                    x=int(centerX-(width /2))
                    y=int(centerY -(height /2))

                    boxes.append([x,y,int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
    idxs= cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)
    if(len(idxs)>0):
        status=list()
        idf=idxs.flatten()
        colse_pair=list()
        center= list()
        co_info=[]

        for i in idf:
            (x,y)=(boxes[i][0],boxes[i][1])
            (w,h)=(boxes[i][2],boxes[i][3])
            cen=[int(x+w/2), int(y+h/2)]
            center.append(cen)
            cv2.circle(frame,tuple(cen),1,(0,0,0),1)
            co_info.append([w,h,cen])
            status.append(0)
        
        for i in range(len(center)):
            for j in range(len(center)):
                g= isclose(co_info[i], co_info[j])

                if g==1:
                    colse_pair.append([co_info[i], co_info[j]])
                    status[i]=1
                    status[j]=1
        
        total_p=len(center)
        high_risk_p=status.count(1)
        safe_p=status.count(0)
        kk=0

        for i in idf:

            (x,y)=[boxes[i][0],boxes[i][1]]
            (w,h)=[boxes[i][2],boxes[i][3]]

            if status[kk]==1:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,150),2)
            elif status[kk]==0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            kk+=1
        
        
            
        cv2.imshow('Social distancing analyser',frame)
        cv2.waitKey(1)
    
    if writer is None:
        fourcc=cv2.VideoWriter_fourcc(*"MJPG")
        writer= cv2.VideoWriter("op2.mp4",fourcc,30,(frame.shape[1],frame.shape[0]),True)
    writer.write(frame)

print("Processing finshed: open output.mp4")
writer.release()
vs.release()


