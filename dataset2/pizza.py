import cv2
import numpy as np
import time


def build_seq():
    for i in range(1158, 1169):
        file_name = "DSC_%d.JPG" % i
        seq.append(file_name)
    
seq = []
result = np.zeros((2160 // 8 * 3, 3840 // 8 * 3, 3))
count = np.zeros((2160 // 8 * 3, 3840 // 8 * 3))
ones = np.ones((2160 // 8 * 3, 3840 // 8 * 3))

kpdetector = cv2.xfeatures2d.SIFT_create() 
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
build_seq()

for idx, image_name in enumerate(seq):

    
    cur_image = cv2.imread(image_name)
    cur_image = cv2.resize(cur_image,(cur_image.shape[1]//8,cur_image.shape[0]//8))
    
    gray = cv2.cvtColor(cur_image, cv2.COLOR_BGR2GRAY)
    kp2 = kpdetector.detect(gray,None)
    dt2 = kpdetector.compute(gray,kp2)[1]

    if idx == 0:
        T      = np.eye(3)
        T[0,2] = (result.shape[1] - cur_image.shape[1]) // 2
        T[1,2] = (result.shape[0] - cur_image.shape[0]) // 2
        result = cv2.warpPerspective(cur_image,T,(result.shape[1],result.shape[0])).astype(np.float)

        t_count= cv2.warpPerspective(ones,T,(result.shape[1],result.shape[0])).astype(np.float)
        count += t_count.astype(np.float)
        disp = result.copy()
        cv2.imshow('stitched image',disp.astype(np.uint8))
    
        kp1 = kp2
        dt1 = dt2
        pre_image = cur_image


    else:

        # Match descriptors.
        matches = bf.match(dt2, dt1)
        print(len(matches))
        
        print('{}, # of matches:{}'.format(idx ,len(matches)))
        
        # Sort in ascending order of distance.
        matches = sorted(matches, key = lambda x:x.distance)
        
        src = []
        dst = []
        for m in matches:
            src.append(kp2[m.queryIdx].pt + (1,))
            dst.append(kp1[m.trainIdx].pt + (1,))
            
        src = np.array(src,dtype=np.float)
        dst = np.array(dst,dtype=np.float)
    
        # find a homography to map src to dst
        A, mask = cv2.findHomography(src, dst, cv2.RANSAC) 
        
        # map to the first frame
        cur_T = T.dot(A)
        warp_img = cv2.warpPerspective(cur_image,cur_T,(result.shape[1],result.shape[0])).astype(np.float)
        t_count  = cv2.warpPerspective(ones,cur_T,(result.shape[1],result.shape[0])).astype(np.float)
        print(t_count.shape)
        result += warp_img
        count += t_count.astype(np.float)

        t_count = count.copy()
        t_count[t_count == 0] = 1
        disp = result.copy()

        
        disp[:,:,0] = result[:,:,0] / t_count
        disp[:,:,1] = result[:,:,1] / t_count
        disp[:,:,2] = result[:,:,2] / t_count
        tmp = disp > 1
        print(np.count_nonzero(tmp))
        
        cv2.imshow('stitched image', disp.astype(np.uint8))
        
        cv2.imshow('matching',cv2.drawMatches(pre_image,kp2,cur_image,kp1,matches[:15], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS))


        
        pre_image = disp.astype(np.uint8)

        gray = cv2.cvtColor(disp.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        """
        cv2.imshow("gray", gray)
        kp1 = kpdetector.detect(gray,None)
        dt1 = kpdetector.compute(gray,kp1)[1]
        T      = np.eye(3)
        """

        kp1 = kp2
        dt1 = dt2
        pre_image = cur_image
        T = cur_T
        


    key = cv2.waitKey(20) & 0xFF
    if key == 27:
        break

cv2.waitKey(0)
"""

# old ---------------------------------------------------------------------

cv2.namedWindow('matching')
cv2.createTrackbar('frame no.','matching',0,total_frame-1,set_frame_number)



while frame_num < total_frame and frame_num < 95: 
    
    if frame_num == 0:
        T      = np.eye(3)
        T[0,2] = result.shape[1]-frame2.shape[1]
        T[1,2] = 0
        result = cv2.warpPerspective(frame2,T,(result.shape[1],result.shape[0])).astype(np.float)
        t_count= cv2.warpPerspective(ones,T,(result.shape[1],result.shape[0])).astype(np.float)
        count += t_count.astype(np.float)
        disp = result.copy()
        cv2.imshow('stitched image',disp.astype(np.uint8))
    
        frame1 = frame2
        kp1 = kp2
        dt1 = dt2
    else:
cv2.waitKey()    
cap.release()
cv2.destroyAllWindows()
"""
