import cv2
import numpy as np

cap = cv2.VideoCapture('./dataset1/hw4_dataset1.mp4')

def pretreat(kpdetector, frame_num):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, cur_frame = cap.read() 
    if ret == False:
        return False, False, False
    cur_frame = cv2.resize(cur_frame,(cur_frame.shape[1]//4,cur_frame.shape[0]//4))
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

def show_image(result, count, out):
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
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frame // 2
    frame_count = 0

    result = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))//4,int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//4*3,3))
    count  = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))//4,int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//4*3))
    ones = np.ones(((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))//4,int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//4)))
    out = cv2.VideoWriter('./result/exercise1.avi',cv2.VideoWriter_fourcc(*'XVID'), 20.0, 
                            ((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//4*3),(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))//4)))
    # create kpdetector
    kpdetector = cv2.xfeatures2d.SIFT_create() 

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    while middle_frame + frame_count < total_frame:
        if frame_count == 0:
            # init
            cur_frame, pre_dt_left, pre_kp_left = pretreat(kpdetector, middle_frame)
            pre_kp_right = pre_kp_left
            pre_dt_right = pre_dt_left

            if type(cur_frame) == type(True):
                break
            # calc loc
            T_left = np.eye(3)
            T_left[0,2] = (result.shape[1] - cur_frame.shape[1]) // 2 - 190
            T_left[1,2] = 0
            T_right = T_left
            # stitch
            count = stitch(cur_frame, result, ones, count, T_left)
            show_image(result, count, out)

        else:
            # stitch left
            cur_frame, dt2, kp2 = pretreat(kpdetector, middle_frame + frame_count)
            if type(cur_frame) == type(True):
                break
            T_right = calc_transfrom(bf, pre_dt_right, dt2, pre_kp_right, kp2, T_right)
            count = stitch(cur_frame, result, ones, count, T_right)
            pre_kp_right = kp2
            pre_dt_right = dt2

            # stith right
            cur_frame, dt2, kp2 = pretreat(kpdetector, middle_frame - frame_count)
            if type(cur_frame) == type(True):
                break
            T_left = calc_transfrom(bf, pre_dt_left, dt2, pre_kp_left, kp2, T_left)
            count = stitch(cur_frame, result, ones, count, T_left)
            pre_kp_left = kp2
            pre_dt_left = dt2

            # show image
            show_image(result, count, out)

        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break
        frame_count += 1

    cv2.waitKey()    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
