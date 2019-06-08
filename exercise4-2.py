import cv2
import numpy as np
import itertools

path = "./dataset2/DSC_11"
out = cv2.VideoWriter('./result/exercise2.avi',cv2.VideoWriter_fourcc(*'XVID'), 20.0, 
                            (3840 // 8 * 3, 2160 // 8 * 3))

def pretreat(kpdetector, image_name): 
    cur_image = cv2.imread(image_name)
    cur_frame = cv2.resize(cur_image,(cur_image.shape[1]//8,cur_image.shape[0]//8))
    gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    kp = kpdetector.detect(gray,None)
    return cur_frame, kpdetector.compute(gray,kp)[1], kp

def calc_transfrom(bf, pre_dt, cur_dt, pre_kp, cur_kp, T):
    matches = bf.match(cur_dt, pre_dt)
    matches = sorted(matches, key = lambda x:x.distance)
            
    src = []
    dst = []
    for m in matches:
        src.append(cur_kp[m.queryIdx].pt + (1,))
        dst.append(pre_kp[m.trainIdx].pt + (1,))
        
    src = np.array(src,dtype=np.float)
    dst = np.array(dst,dtype=np.float)
            
    # find a homography to map src to dst
    A, mask = cv2.findHomography(src, dst, cv2.RANSAC) 
    T = T.dot(A)
    return T

def show_image(result, count):
    t_count = count.copy()
    t_count[t_count == 0] = 1

    disp = result.copy()
    disp[:,:,0] = result[:,:,0] / t_count
    disp[:,:,1] = result[:,:,1] / t_count
    disp[:,:,2] = result[:,:,2] / t_count

    cv2.imshow('stitched image',disp.astype(np.uint8))
    out.write(disp.astype(np.uint8))


def stitch(cur_frame, result, ones, count, T):
    warp_img = cv2.warpPerspective(cur_frame,T,(result.shape[1],result.shape[0])).astype(np.float)
    t_count  = cv2.warpPerspective(ones,T,(result.shape[1],result.shape[0])).astype(np.float)
    result += warp_img
    count = t_count.astype(np.float) + count
    return count

def main():
    sample_image = cv2.imread(path + "58" + ".JPG")
    

    # create kpdetector
    kpdetector = cv2.xfeatures2d.SIFT_create() 

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)


    result = np.zeros((2160 // 8 * 3, 3840 // 8 * 3, 3))
    count = np.zeros((2160 // 8 * 3, 3840 // 8 * 3))
    ones = np.ones((2160 // 8 , 3840 // 8 ))
    frame_count = 0
    for frame_num in range(58, 69):
        if frame_num == 58:
            # init
            cur_frame, pre_dt, pre_kp = pretreat(kpdetector, path + str(frame_num) + ".JPG")

            if type(cur_frame) == type(True):
                break
            # calc loc
            T = np.eye(3)
            T[0,2] = (result.shape[1] - cur_frame.shape[1]) // 2
            T[1,2] = (result.shape[0] - cur_frame.shape[0]) // 2
            # stitch
            count = stitch(cur_frame, result, ones, count, T)
            show_image(result, count)
            
        else:
            # stitch
            cur_frame, dt2, kp2 = pretreat(kpdetector, path + str(frame_num) + ".JPG")
            if type(cur_frame) == type(True):
                break
            T = calc_transfrom(bf, pre_dt, dt2, pre_kp, kp2, T)
            count = stitch(cur_frame, result, ones, count, T)
            pre_kp = kp2
            pre_dt = dt2
            show_image(result, count)

        key = cv2.waitKey(20) & 0xFF
        if key is 27:
            break
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
