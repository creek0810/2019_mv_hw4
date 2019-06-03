import cv2
import numpy as np


result = np.zeros((2160 //4*4,3840//4*3,3))
count  = np.zeros((2160 //4*4,3840//4*3))
ones = np.ones((2160 //4 ,3840//4))


kpdetector = cv2.xfeatures2d.SIFT_create() 
#kpdetector = cv2.AKAZE_create()

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
frame_num = 0
seq = []
#tmp = [1162, 1163, 1161, 1164, 1165, 1166, 1167, 1169, 1168, 1160, 1159]
tmp = [1159, 1169, 1168, 1167]

for i in tmp:
    file_name = "DSC_%d.JPG" % i
    seq.append(file_name)

for i in seq:
    frame2 = cv2.imread(i)
      
    
    frame2 = cv2.resize(frame2,(frame2.shape[1]//4,frame2.shape[0]//4))
    
#kp1, dt1 = kpdetector.detectAndCompute(frame1,None)
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    kp2 = kpdetector.detect(gray,None)
    dt2 = kpdetector.compute(gray,kp2)[1]
    if frame_num == 0:
        T      = np.eye(3)
        T[0,2] = (result.shape[1]-frame2.shape[1]) // 2
        T[1,2] = (result.shape[0]-frame2.shape[0]) // 2
        result = cv2.warpPerspective(frame2,T,(result.shape[1],result.shape[0])).astype(np.float)
        t_count= cv2.warpPerspective(ones,T,(result.shape[1],result.shape[0])).astype(np.float)
        count += t_count.astype(np.float)
        disp = result.copy()
        cv2.imshow('stitched image',disp.astype(np.uint8))
    
        frame1 = frame2
        kp1 = kp2
        dt1 = dt2
    else:
        # Match descriptors.
        #matches = bf.knnMatch(dt2,dt1, k=2)
        #print('{}, # of matches:{}'.format(frame_num,len(matches)))

        # Sort in ascending order of distance.
        #matches = sorted(matches, key = lambda x:x.distance)
        FLANN_INDEX_KDTREE=0   #建立FLANN匹配器的参数
        indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5) #配置索引，密度树的数量为5
        searchParams=dict(checks=50)    #指定递归次数
        #FlannBasedMatcher：是目前最快的特征匹配算法（最近邻搜索）
        flann=cv2.FlannBasedMatcher(indexParams,searchParams)  #建立匹配器
        matches=flann.knnMatch(dt1,dt2,k=2)  #得出匹配的关键点

      
        good=[]
        for m,n in matches:
            if m.distance < 0.7*n.distance: #如果第一个邻近距离比第二个邻近距离的0.7倍小，则保留
                good.append(m)
        dst = np.array([ kp1[m.queryIdx].pt for m in good])    #查询图像的特征描述子索引
        src = np.array([ kp2[m.trainIdx].pt for m in good])

        """
        src = []
        dst = []
        
        for m in good:
            src.append(kp2[m.queryIdx].pt + (1,))
            dst.append(kp1[m.trainIdx].pt + (1,))
        """     

        
        
        src = np.array(src,dtype=np.float)
        dst = np.array(dst,dtype=np.float)
    
        # find a homography to map src to dst
        A, mask = cv2.findHomography(src, dst, cv2.RANSAC) 
        
        # map to the first frame
        T = T.dot(A)
        warp_img = cv2.warpPerspective(frame2,T,(result.shape[1],result.shape[0])).astype(np.float)
        t_count  = cv2.warpPerspective(ones,T,(result.shape[1],result.shape[0])).astype(np.float)
        result+= warp_img
        count += t_count.astype(np.float)

        t_count= count.copy()
        t_count[t_count == 0] = 1
        disp = result.copy()
        
        disp[:,:,0] = result[:,:,0] / t_count
        disp[:,:,1] = result[:,:,1] / t_count
        disp[:,:,2] = result[:,:,2] / t_count
 
        cv2.imshow('stitched image',disp.astype(np.uint8))
   
        #cv2.imshow('matching',cv2.drawMatches(frame2,kp2,frame1,kp1,good[:], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS))
        #cv2.imshow('matching',cv2.drawMatchesKnn(frame2,kp2,frame1,kp1,good, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS))
        
        """
        frame1 = frame2
        kp1 = kp2
        dt1 = dt2
        """
        frame1 = disp.astype(np.uint8)
        gray = cv2.cvtColor(disp.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        kp1 = kpdetector.detect(gray,None)
        dt1 = kpdetector.compute(gray,kp1)[1]
        T      = np.eye(3)

    key = cv2.waitKey(20) & 0xFF
    if key == 27:
        break
    frame_num += 1
cv2.waitKey()    
cap.release()
cv2.destroyAllWindows()
