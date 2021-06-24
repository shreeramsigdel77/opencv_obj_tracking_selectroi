import cv2
import numpy as np

def drawBox(img, bbox):
    x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv2.rectangle(img,(x,y),((x+w),(y+h)),(0,0,255),3,1)
    cv2.putText(img,"Tracking", (50,60),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)


#load video from camera

cap = cv2.VideoCapture(0)

#tracker for opencv

tracker = cv2.legacy.TrackerMOSSE_create() 
# tracker = cv2.TrackerCRST.create()
success, img = cap.read()
print("success",success)

# bbox = cv2.selectROIs("Tracking",img,False,False)
bbox = cv2.selectROI("Tracking",img,False,False)

# cv2.destroyWindow('Tracking')

# if len(bbox)>0:
#     bbox = bbox[-1]
# bbox = np.array(bbox)


tracker.init(img,bbox)


while True:
    timer = cv2.getTickCount()
    
    success, img = cap.read()
    success,bbox =tracker.update(img)   

    if success:
        drawBox(img,bbox)
    else:
        cv2.putText(img,"Lost", (50,60),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

    fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    cv2.putText(img,str(int(fps)), (50,50),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)





    cv2.imshow("Tracking", img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        cv2.destroyAllWindows()
        break