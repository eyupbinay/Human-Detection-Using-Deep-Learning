# -*- coding: utf-8 -*-
"""


@project: human detection using deep learning 

"""

import cv2
from persondetection import DetectorAPI


# initialize the variables
max_count2 = 0
framex2 = []
county2 = []
max2 = []
avg_acc2_list = []
max_avg_acc2_list = []
max_acc2 = 0
max_avg_acc2 = 0
Numb_of_peraons = 0

cv2.startWindowThread()

# open webcam video stream
#cap = cv2.VideoCapture("testVedio.mp4")
cap = cv2.VideoCapture(0)
odapi = DetectorAPI()
threshold = 0.95
# the output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640, 480))
x2 = 0
old_person = 0
num_of_persons = 0
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret: 
        # resizing for faster detection
        img = cv2.resize(frame, (640, 480))
        boxes, scores, classes, num = odapi.processFrame(img)
        person = 0
        acc = 0
        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] >= threshold:
                box = boxes[i]
                person += 1
                cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)  # cv2.FILLED
                cv2.putText(img, f'P{person, round(scores[i],2)}', (box[1]-30, box[0]-8), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1 )#(75,0,130),
                acc+=scores[i]
                
                if(scores[i]>max_acc2):
                    max_acc2 = scores[i]   
        x2+=1
        currentframe = x2
        if(person>max_count2):
            max_count2 = person  
        if old_person == 0 and person == 1:
           county2.append(person)
           num_of_persons +=1
           print(' county2')    
           print(county2)
           print('num_of_persons')
           print(num_of_persons)
           #Take image and save it in (data) file already created
           name = './data/frame' + str(currentframe) + '.jpg'
           print ('Creating...' + name)
           cv2.imwrite(name, frame)
        old_person = person
        framex2.append(x2)
        if(person>=1):
            avg_acc2_list.append(acc/person)
            if((acc/person)>max_avg_acc2):
                max_avg_acc2 = (acc/person)
        else:
            avg_acc2_list.append(acc)

        cv2.imshow("Human Detection from Video", img)          
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #out automaticlly when the video end    
    else: 
        break

cap.release()
out.release()
cv2.destroyAllWindows()
cv2.waitKey(0)
