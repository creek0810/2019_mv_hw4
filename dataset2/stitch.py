import cv2
import numpy as np
import time
seq = []
tmp = [1162,1161,1163, 1160,1164,1159,1165,1158,1166,1167,1168]
for i in tmp:
    file_name = "DSC_%d.JPG" % i
    seq.append(file_name)
    


IMAGE = cv2.imread(seq[0])

result = np.zeros((int(IMAGE.shape[0])//4*4,int(IMAGE.shape[1])//4*3,3))
count  = np.zeros((int(IMAGE.shape[0])//4*4,int(IMAGE.shape[1])//4*3))
ones = np.ones(((int(IMAGE.shape[0])//4*4,int(IMAGE.shape[1])//4)))



kpdetector = cv2.xfeatures2d.SIFT_create() 
#kpdetector = cv2.AKAZE_create()

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

for idx, img_name in enumerate(seq):
    frame2 = cv2.imread(img_name)
    
       
    
    frame2 = cv2.resize(frame2,(frame2.shape[1]//4,frame2.shape[0]//4))
    
#kp1, dt1 = kpdetector.detectAndCompute(frame1,None)
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    kp2 = kpdetector.detect(gray,None)
    dt2 = kpdetector.compute(gray,kp2)[1]
    if idx == 0:
        T_left      = np.eye(3)
        T_left[0,2] = (result.shape[1]-frame2.shape[1]) // 2
        T_left[1,2] = (result.shape[0]-frame2.shape[0]-1500) // 2
        result = cv2.warpPerspective(frame2,T_left,(result.shape[1],result.shape[0])).astype(np.float)
        t_count= cv2.warpPerspective(ones,T_left,(result.shape[1],result.shape[0])).astype(np.float)
        count += t_count.astype(np.float)
        disp = result.copy()
        cv2.imshow('stitched image',disp.astype(np.uint8))
    
        frame1_left = frame2
        frame1_right = frame2
        kp1_left = kp2
        kp1_right = kp2
        dt1_left = dt2
        dt1_right = dt2
        T_right = T_left
    else:
        #right
        if idx % 2 == 1 or idx == 10:     

            # Match descriptors.
            matches = bf.match(dt2,dt1_left)
            print('{}, # of matches:{}'.format(idx,len(matches)))

            # Sort in ascending order of distance.
            matches = sorted(matches, key = lambda x:x.distance)
            
            src = []
            dst = []
            for m in matches:
                src.append(kp2[m.queryIdx].pt + (1,))
                dst.append(kp1_left[m.trainIdx].pt + (1,))
                
            src = np.array(src,dtype=np.float)
            dst = np.array(dst,dtype=np.float)
        
            # find a homography to map src to dst
            A, mask = cv2.findHomography(src, dst, cv2.RANSAC) 
            
            # map to the first frame
            T_left = T_left.dot(A)
            warp_img = cv2.warpPerspective(frame2,T_left,(result.shape[1],result.shape[0])).astype(np.float)
           

            rows, cols = result.shape[:2]

            for col in range(0, cols):
                if result[:, col].any() and warp_img[:, col].any():
                    left = col
                    break
            for col in range(cols-1, 0, -1):
                if result[:, col].any() and warp_img[:, col].any():
                    right = col
                    break

            res = np.zeros([rows, cols, 3], np.uint8)
            for row in range(0, rows):
                for col in range(0, cols):
                    if not result[row, col].any():
                        res[row, col] = warp_img[row, col]
                    elif not warp_img[row, col].any():
                        res[row, col] = result[row, col]
                    else:
                        srcImgLen = float(abs(col - left))
                        testImgLen = float(abs(col - right))
                        alpha = srcImgLen / (srcImgLen + testImgLen)
                        res[row, col] = np.clip(result[row, col] * (1-alpha) + warp_img[row, col] * alpha, 0, 255)


            cv2.imshow('stitched image', res)

            result = res.astype(np.uint8)



            #result = disp.astype(np.uint8)

            #cv2.imshow('stitched image', disp.astype(np.uint8))
       
            cv2.imshow('matching',cv2.drawMatches(frame2,kp2,frame1_left,kp1_left,matches[:15], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS))
            


            frame1_left = cv2.resize(result,(result.shape[1],result.shape[0]))
        
        #kp1, dt1 = kpdetector.detectAndCompute(frame1,None)
            """
            gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            kp1 = kpdetector.detect(gray,None)
            dt1 = kpdetector.compute(gray,kp1)[1]
            T      = np.eye(3)
            """
            kp1_left = kp2
            dt1_left = dt2
            #time.sleep(5)
        else:

            # Match descriptors.
            matches = bf.match(dt2,dt1_right)
            print('{}, # of matches:{}'.format(idx,len(matches)))

            # Sort in ascending order of distance.
            matches = sorted(matches, key = lambda x:x.distance)
            
            src = []
            dst = []
            for m in matches:
                src.append(kp2[m.queryIdx].pt + (1,))
                dst.append(kp1_right[m.trainIdx].pt + (1,))
                
            src = np.array(src,dtype=np.float)
            dst = np.array(dst,dtype=np.float)
        
            # find a homography to map src to dst
            A, mask = cv2.findHomography(src, dst, cv2.RANSAC) 
            
            # map to the first frame
            T_right = T_right.dot(A)
            warp_img = cv2.warpPerspective(frame2,T_right,(result.shape[1],result.shape[0])).astype(np.float)
           

            rows, cols = result.shape[:2]

            for col in range(0, cols):
                if result[:, col].any() and warp_img[:, col].any():
                    left = col
                    break
            for col in range(cols-1, 0, -1):
                if result[:, col].any() and warp_img[:, col].any():
                    right = col
                    break

            res = np.zeros([rows, cols, 3], np.uint8)
            for row in range(0, rows):
                for col in range(0, cols):
                    if not result[row, col].any():
                        res[row, col] = warp_img[row, col]
                    elif not warp_img[row, col].any():
                        res[row, col] = result[row, col]
                    else:
                        srcImgLen = float(abs(col - left))
                        testImgLen = float(abs(col - right))
                        alpha = srcImgLen / (srcImgLen + testImgLen)
                        res[row, col] = np.clip(result[row, col] * (1-alpha) + warp_img[row, col] * alpha, 0, 255)


            cv2.imshow('stitched image', res)

            result = res.astype(np.uint8)



            #result = disp.astype(np.uint8)

            #cv2.imshow('stitched image', disp.astype(np.uint8))
       
            cv2.imshow('matching',cv2.drawMatches(frame2,kp2,frame1_right,kp1_right,matches[:15], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS))
            


            frame1_right = cv2.resize(result,(result.shape[1],result.shape[0]))
        
        #kp1, dt1 = kpdetector.detectAndCompute(frame1,None)
            """
            gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            kp1 = kpdetector.detect(gray,None)
            dt1 = kpdetector.compute(gray,kp1)[1]
            T      = np.eye(3)
            """
            kp1_right = kp2
            dt1_right = dt2
            #time.sleep(5)

    key = cv2.waitKey(20) & 0xFF
    if key == 27:
        break
cv2.waitKey()    
cap.release()
cv2.destroyAllWindows()
