import cv2

l = "l_bgr_32_sadwin_13_sigma_1.0_lambda_10000.0_time_9.0ms.png"
r = "r_bgr_32_sadwin_13_sigma_1.0_lambda_10000.0_time_9.0ms.png"

bgr_l = cv2.imread(l)
bgr_r = cv2.imread(r)

rgb_l = cv2.cvtColor(bgr_l, cv2.COLOR_BGR2RGB)
rgb_r = cv2.cvtColor(bgr_r, cv2.COLOR_BGR2RGB)

cv2.imwrite("l_rgb_32_sadwin_13_sigma_1.0_lambda_10000.0_time_9.0ms.png", rgb_l)
cv2.imwrite("r_rgb_32_sadwin_13_sigma_1.0_lambda_10000.0_time_9.0ms.png", rgb_r)
