import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt

def form_transf(R,t):
    """
        Transformation matrix
    """
    T = np.eye(4,dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = t
    return T


video_cap = cv2.VideoCapture('nopatos.mp4') #480x848
ret, last_frame = video_cap.read()

current_frame = None
while video_cap.isOpened():
    ret,frame = video_cap.read()

    frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Video sin patos
    # frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[0:250,200:600] # Video con patos

    query_img = frame_bw.copy()
    train_img = last_frame.copy() 
    # f = cv2.imread('.jpg', cv2.IMREAD_GRAYSCALE)
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(frame_bw,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(frame_bw, kp)

    queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img,None)
    trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img,None)
    # draw only keypoints location,not size and orientation

    matcher = cv2.BFMatcher()
    matches = matcher.match(queryDescriptors,trainDescriptors)
    # print(queryDescriptors)

    img2 = cv2.drawKeypoints(frame_bw, kp, None, color=(0,255,255), flags=0)

    final_image= cv2.drawMatches(query_img, queryKeypoints, train_img, trainKeypoints, matches[:20], None, flags=2)
    final_image = cv2.resize(final_image, (1000,650))
    cv2.imshow('orbi test',final_image)
    if cv2.waitKey(1) == ord('q'):
        break
    last_frame = frame_bw
video_cap.release()
cv2.destroyAllWindows()