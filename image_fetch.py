# import the opencv library
import cv2


from math import atan2, cos, sin, sqrt, pi
import numpy as np

def drawAxis(img, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)

    ## [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 1, cv2.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 1, cv2.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 1, cv2.LINE_AA)
    ## [visualization1]

def getOrientation(pts, img, mean):
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]

    # Perform PCA analysis
    mean = np.empty((0))

    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    ## [pca]

    ## [visualization]
    # Draw the principal components
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (255, 255, 0), 1)
    drawAxis(img, cntr, p2, (0, 0, 255), 5)

    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    ## [visualization]

    # Label with the rotation angle
    # label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
    # textbox = cv2.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
    # cv2.putText(img, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    return angle

def getLine(cnt, img):
    rows,cols = img.shape[:2]
    [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)



def count_black_pixels(image, center, r):

    # Step 1: Create a mask with the same dimensions as the image, initially all zeros
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Step 2: Draw a filled circle (white) on the mask at the specified position and radius
    cv2.circle(mask, center, r, (255), thickness=-1)

    # Invert the mask if you want the circle to be black and the rest white
    image = cv2.bitwise_not(image)

    cv2.imshow('mask', mask)

    # Apply the mask to the image (if needed, depending on your specific problem)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow('masked_image', masked_image)

    # Step 3: Count the black pixels. If you've inverted the mask, count directly.
    # If not, you'll need to adjust based on your scenario.
    # Here, counting directly as if we've inverted the mask.
    # Note: Assuming the image is grayscale. If it's color, convert it to grayscale first.
    num_black_pixels = cv2.countNonZero(masked_image)

    return num_black_pixels

# define a video capture object
vid = cv2.VideoCapture(0)

calibration = True

while(True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # # using cv2.Canny() for edge detection.
    # edge_detect = cv2.Canny(frame, 120, 400)
    # cv2.imshow('Edge detect', edge_detect)w

    # converting image into grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # setting threshold of gray image
    #_, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    #ret, th1 = cv.threshold(frame,127,255,cv.THRESH_BINARY)

    # thre = cv.adaptiveThreshold(frame,255,cv.ADAPTIVE_THRESH_MEAN_C,\
    #                            cv.THRESH_BINARY,11,2)


    blur = cv2.GaussianBlur(gray, (5,5),0)
    ret2,threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    # blur = cv2.GaussianBlur(gray, (5,5),0)
    # ret3,threshold = cv2.threshold(blur, 100, 155, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #                                   cv2.THRESH_BINARY, 11,2)

    cv2.imshow('Thresh', threshold)

    # using a findContours() function
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    i = 0

    valid_slices = []

    # list for storing names of shapes
    for idx, contour in enumerate(contours):

        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(
            contour, 0.05 * cv2.arcLength(contour, True), True)


        # finding center point of shape
        M = cv2.moments(contour)

        if cv2.contourArea(contour) < 1500:
            continue

        # using drawContours() function
        cv2.drawContours(frame, [contour], 0, (0, 0, 255), 1)


        if M['m00'] > 0.0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])

        # putting shape name at center of each shape
        if 2 <= len(approx) <= 5 :
            # cv2.putText(frame, f'{x}, {y}', (x, y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.circle(frame, (x,y), radius=10, color=(0, 0, 255), thickness=-1)

            valid_slices.append(contour)

        #getOrientation(contour, frame, (x,y))
        #getLine(contour, frame)


        # elif len(approx) == 4:
        #     cv2.putText(frame, 'Quadrilateral', (x, y),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # elif len(approx) == 5:
        #     cv2.putText(frame, 'Pentagon', (x, y),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # elif len(approx) == 6:
        #     cv2.putText(frame, 'Hexagon', (x, y),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # else:
        #     cv2.putText(frame, 'circle', (x, y),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    try:
        if calibration:
            all_points = np.vstack(valid_slices)

            (c_x, c_y), radius = cv2.minEnclosingCircle(all_points)
            center = (int(c_x), int(c_y))
            radius = int(radius)
    except:
        pass

    cv2.circle(frame, center, radius, (0, 255, 0), 2)

    cv2.circle(frame, center, radius=1, color=(0, 255, 0), thickness=-1)

    area = count_black_pixels(threshold, center, radius)

    cv2.putText(frame, 'Area %d C: %d' % (area, calibration), (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    #ellipse = cv2.fitEllipse(all_points)
    #cv2.ellipse(frame, ellipse, (0,255,0), 1)


    # displaying the image after drawing contours
    cv2.imshow('shapes', frame)


    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('c'):
        calibration = not calibration


# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
