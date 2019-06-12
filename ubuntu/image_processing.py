import matplotlib.pyplot as plt
import numpy as np
import cv2

# load
l_disp = cv2.imread("left_disp.jpg")
r_disp = cv2.imread("right_disp.jpg")

numDisparities = 32
cv2.normalize(l_disp,  l_disp, 0, numDisparities-1, cv2.NORM_MINMAX)
cv2.normalize(r_disp,  r_disp, 0, numDisparities-1, cv2.NORM_MINMAX)

l_disp = l_disp[:,:,0].astype(np.float32)
r_disp = r_disp[:,:,0].astype(np.float32)

# compute variance maps
rad = 5

l_avg = cv2.boxFilter(l_disp, ddepth=cv2.CV_32F, ksize=(2*rad+1, 2*rad+1), normalize=True)
r_avg = cv2.boxFilter(r_disp, ddepth=cv2.CV_32F, ksize=(2*rad+1, 2*rad+1), normalize=True)

l_sq_avg = cv2.sqrBoxFilter(l_disp, ddepth=cv2.CV_32F, ksize=(2*rad+1, 2*rad+1), normalize=True)
r_sq_avg = cv2.sqrBoxFilter(r_disp, ddepth=cv2.CV_32F, ksize=(2*rad+1, 2*rad+1), normalize=True)

l_var = l_sq_avg - np.multiply(l_avg, l_avg)
r_var = r_sq_avg - np.multiply(r_avg, r_avg)

# confidence maps
roll_off = 0.001

l_conf = np.maximum(1.0 - roll_off*l_var, 0.)
r_conf = np.maximum(1.0 - roll_off*r_var, 0.)

# left right consistency
lrc = np.zeros(l_conf.shape)
thresh = 24

for i in range(r_conf.shape[0]):
    for j in range(l_conf.shape[1]):
        rj = j - int(int(l_disp[i, j])>>4) # bitshift by 4, divides by powers of 2 (here: 2**4)
        if rj >= 0 and rj < r_conf.shape[1]:
            disp = abs(l_disp[i, j] + r_disp[i, rj])
            if disp < thresh:
                lrc[i, j] = min(l_conf[i, j], r_conf[i, rj])
            else:
                lrc[i, j] = 0.

# disp multiply conf
disp_mul_conf = np.multiply(l_disp, lrc)
        
# normalize
cv2.normalize(l_avg,  l_avg, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(r_avg,  r_avg, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(l_sq_avg,  l_sq_avg, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(r_sq_avg,  r_sq_avg, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(l_var,  l_var, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(r_var,  r_var, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(l_conf,  l_conf, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(r_conf,  r_conf, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(lrc,  lrc, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(disp_mul_conf, disp_mul_conf, 0, 255, cv2.NORM_MINMAX)

# save
cv2.imwrite("l_avg.jpg", l_avg)
cv2.imwrite("r_avg.jpg", r_avg)
cv2.imwrite("l_sq_avg.jpg", l_sq_avg)
cv2.imwrite("r_sq_avg.jpg", r_sq_avg)
cv2.imwrite("l_var.jpg", l_var)
cv2.imwrite("r_var.jpg", r_var)
cv2.imwrite("l_conf.jpg", l_conf)
cv2.imwrite("r_conf.jpg", r_conf)
cv2.imwrite("lrc.jpg", lrc)
cv2.imwrite("disp_mul_conf.jpg", disp_mul_conf)

