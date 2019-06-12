import matplotlib.pyplot as plt
import numpy as np
import cv2

# load
l_disp = cv2.imread("left_disp.jpg")
r_disp = cv2.imread("right_disp.jpg")

numDisparities = 32
blockSize = 13

cv2.normalize(l_disp,  l_disp, 0, numDisparities-1, cv2.NORM_MINMAX)
cv2.normalize(r_disp,  r_disp, 0, numDisparities-1, cv2.NORM_MINMAX)

l_disp = l_disp[:,:,0].astype(np.float32)
r_disp = r_disp[:,:,0].astype(np.float32)

l_offs = int(numDisparities) + int(blockSize/2.)
r_offs = t_offs = b_offs = int(blockSize/2.)

l_disp_val = l_disp[t_offs:l_disp.shape[0]-b_offs,l_offs:l_disp.shape[1]-r_offs]
r_disp_val =  l_disp[t_offs:r_disp.shape[0]-b_offs,l_offs:r_disp.shape[1]-r_offs]

#valid_disp_ROI = cv2.rectangle(l_offs, t_offs, l_disp.shape[1]-l_offs-r_offs, l_disp.shape[0]-t_offs-b_offs)

# compute variance maps
rad = 5

l_avg = np.zeros(l_disp.shape)
r_avg = np.zeros(r_disp.shape)

l_avg[t_offs:l_disp.shape[0]-b_offs,l_offs:l_disp.shape[1]-r_offs] = cv2.boxFilter(l_disp_val, ddepth=cv2.CV_32F, ksize=(2*rad+1, 2*rad+1), normalize=True)
r_avg[t_offs:r_disp.shape[0]-b_offs,l_offs:r_disp.shape[1]-r_offs] = cv2.boxFilter(r_disp_val, ddepth=cv2.CV_32F, ksize=(2*rad+1, 2*rad+1), normalize=True)

l_sq_avg = np.zeros(l_disp.shape)
r_sq_avg = np.zeros(r_disp.shape)

l_sq_avg[t_offs:l_disp.shape[0]-b_offs,l_offs:l_disp.shape[1]-r_offs] = cv2.sqrBoxFilter(l_disp_val, ddepth=cv2.CV_32F, ksize=(2*rad+1, 2*rad+1), normalize=True)
r_sq_avg[t_offs:r_disp.shape[0]-b_offs,l_offs:r_disp.shape[1]-r_offs] = cv2.sqrBoxFilter(r_disp_val, ddepth=cv2.CV_32F, ksize=(2*rad+1, 2*rad+1), normalize=True)

l_var = np.zeros(l_disp.shape)
r_var = np.zeros(r_disp.shape)

l_var = l_sq_avg - np.multiply(l_avg, l_avg)
r_var = r_sq_avg - np.multiply(r_avg, r_avg)

# confidence maps
roll_off = 0.001

l_conf = np.maximum(1.0 - roll_off*l_var, 0.)
r_conf = np.maximum(1.0 - roll_off*r_var, 0.)

# left right consistency
lrc = np.zeros(l_disp.shape)
thresh = 24

l_offs_right = l_disp.shape[1] - (l_offs + l_disp.shape[1]-r_offs)
t_offs_right = t_offs

for i in range(t_offs, l_disp.shape[0]-b_offs):
    for j in range(l_offs, l_disp.shape[1]-r_offs):
        rj = j - int(int(l_disp[i, j])) # bitshift by 4, divides by powers of 2 (here: 2**4)
        if rj >= l_offs_right and rj < l_disp.shape[1] - -l_offs - r_offs:
            disp = abs(l_disp[i, j] - r_disp[i, rj])
            if disp < thresh:
                lrc[i, j] = min(l_conf[i, j], r_conf[i, rj])
            else:
                lrc[i, j] = 0.

lrc *= 255.

# disp multiply conf
disp_mul_conf = np.multiply(l_disp, lrc)

# wls
l_gra = cv2.imread("gra_left.jpg")
l_gra = l_gra[:,:,0]

disp_mul_conf_filtered = np.zeros(l_disp.shape)
lrc_filtered = np.zeros(l_disp.shape)

wls = cv2.ximgproc.createFastGlobalSmootherFilter(l_gra[t_offs:l_disp.shape[0]-b_offs,l_offs:l_disp.shape[1]-r_offs], 1.e4, 1.)
disp_mul_conf_filtered[t_offs:l_disp.shape[0]-b_offs,l_offs:l_disp.shape[1]-r_offs] = wls.filter(disp_mul_conf[t_offs:l_disp.shape[0]-b_offs,l_offs:l_disp.shape[1]-r_offs].astype(np.float32))
lrc_filtered[t_offs:l_disp.shape[0]-b_offs,l_offs:l_disp.shape[1]-r_offs] = wls.filter(lrc[t_offs:l_disp.shape[0]-b_offs,l_offs:l_disp.shape[1]-r_offs].astype(np.float32))

wls_filtered = np.multiply(disp_mul_conf_filtered, 1./(lrc_filtered+1.e-43))

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
cv2.normalize(wls_filtered,  wls_filtered, 0, 255, cv2.NORM_MINMAX)

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
cv2.imwrite("wls_filtered.jpg", wls_filtered)

