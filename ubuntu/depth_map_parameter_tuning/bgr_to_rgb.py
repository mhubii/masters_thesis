import cv2

l = "28_07_19_wls_measurements/l_bgr_32_sadwin_13_sigma_0.1_lambda_100.0_time_25.0ms.png"
r = "28_07_19_wls_measurements/r_bgr_32_sadwin_13_sigma_0.1_lambda_100.0_time_25.0ms.png"

bgr_l = cv2.imread(l)
bgr_r = cv2.imread(r)

rgb_l = cv2.cvtColor(bgr_l, cv2.COLOR_BGR2RGB)
rgb_r = cv2.cvtColor(bgr_r, cv2.COLOR_BGR2RGB)

cv2.imwrite("l_rgb_32_sadwin_13_sigma_0.1_lambda_100.0_time_25.0ms.png", rgb_l)
cv2.imwrite("r_rgb_32_sadwin_13_sigma_0.1_lambda_100.0_time_25.0ms.png", rgb_r)
