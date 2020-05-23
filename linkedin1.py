import time
import cv2
import numpy as np


confid=0.5
thresh= 0.5

vid_path="./videos/walking_people.mp4"

def calibrated_dist(p1,p2):
    return ((p1[0]-p2[0])**2+550 /((p1[1]+p2[1])/2)*(p1[1]-p2[1])**2)**0.5

def isclose(p1,p2):
    c_d=calibrated_dist(p1,p2)
    calib=(p1[1]+p2[1]/2)
    if 0<c_d<0.15*calib:
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
        q=W
    
    frame= frame[0:H, 200:q]
    (H,W)=frame.shape[:2]
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
        dist =list()

        for i in idf:
            (x,y)=(boxes[i][0],boxes[i][1])
            (w,h)=(boxes[i][2],boxes[i][3])
            center.append([int(x+w/2), int(y+h/2)])
            status.append(0)
        
        for i in range(len(center)):
            for j in range(len(center)):
                g= isclose(center[i], center[j])

                if g==1:
                    colse_pair.append([center[i],center[j]])
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
        
        for h in colse_pair:
            cv2.line(frame,tuple(h[0]),tuple(h[1]),(0,0,255),2)
            
        cv2.imshow('Social distancing analyser',frame)
        cv2.waitKey(1)
    
    if writer is None:
        fourcc=cv2.VideoWriter_fourcc(*"MJPG")
        writer= cv2.VideoWriter("o.mp4",fourcc,30,(frame.shape[1],frame.shape[0]),True)
    writer.write(frame)

print("Processing finshed: open output.mp4")
writer.release()
vs.release()


