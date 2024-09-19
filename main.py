import cv2 
from HandDetector import HandDetector 

detector = HandDetector(maxHands=1, detectionCon=0.8) 

def printText(img, text):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale              = 3
    fontColor              = (128,255,255)
    lineType               = 5

    cv2.putText(img,text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

def main():
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret: 
                break
            
            frame = cv2.flip(frame,1)
            hands = detector.findHands(frame, draw=True) 
            if len(hands) > 0: 
                landmarks = hands[0]
                if landmarks:
                    try:
                        fingerup = detector.fingersUp(landmarks[0])
                        printText (frame, str(sum(fingerup)))
                    finally:
                        pass
            cv2.imshow('frame',frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()          