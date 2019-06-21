from scipy import ndimage, misc

left = misc.imread("gra_left.jpg", mode='L')
right = misc.imread("gra_right.jpg", mode='L')

left = ndimage.sobel(left, axis=1)
right = ndimage.sobel(right, axis=1)

misc.imsave("sobel_left.jpg", left)
misc.imsave("sobel_right.jpg", right)
