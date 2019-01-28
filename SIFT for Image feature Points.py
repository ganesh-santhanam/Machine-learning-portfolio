import cv2
import numpy as np
import math
import scipy.stats as st
import scipy


def scalingandblurringoutput(image,xscale,yscale,kernlen,nsig):
scalingimage = cv2.resize(image, (0,0), fx=xscale, fy=yscale)
rows,columns= scalingimage.shape
print (rows,columns)
gauss_blur_filter = [[0 for x in range(5)] for y in range(5)]
dummy_image_x = [[0 for x in range(columns)] for y in range(rows)]
new_image = [[0 for x in range(columns + 6)] for y in range(rows + 6 )]
pad_image = np.asarray(scalingimage)
for i in range(rows+3):
for j in range(columns+3):
if i == 0 or i== 1 or i == rows+1 or i == rows+2 or i == rows+3:
  new_image[i][j] = 0
elif j == 0 or j == 1 or j == columns+1 or j == columns+2 or j == columns+3:
  new_image[i][j] = 0
else:
  new_image[i][j] = pad_image[i-3][j-3]
gauss_blur_filter = np.asarray(gaussian_blur_matrix(kernlen,nsig))

    for i in range(rows):
      for j in range(columns):
          dummy_image_x[i][j] = new_image[i][j] * gauss_blur_filter[0][0] + new_image[i][j+1] * gauss_blur_filter[0][1] + new_image[i][j+2] * gauss_blur_filter[0][2] + new_image[i][j+3] * gauss_blur_filter[0][3] +new_image[i][j+4] * gauss_blur_filter[0][4] + new_image[i][j+5] * gauss_blur_filter[0][5] + new_image[i][j+6] * gauss_blur_filter[0][6] + new_image[i+1][j] * gauss_blur_filter[1][0] + new_image[i+1][j+1] * gauss_blur_filter[1][1] + new_image[i+1][j+2] * gauss_blur_filter[1][2] + new_image[i+1][j+3] * gauss_blur_filter[1][3] + new_image[i+1][j+4] * gauss_blur_filter[1][4] + new_image[i+1][j+5] * gauss_blur_filter[1][5] + new_image[i+1][j+6] * gauss_blur_filter[1][6]+ new_image[i+2][j] * gauss_blur_filter[2][0] + new_image[i+2][j+1] * gauss_blur_filter[2][1] + new_image[i+2][j+2] * gauss_blur_filter[2][2] + new_image[i+2][j+3] * gauss_blur_filter[2][3] + new_image[i+2][j+4] * gauss_blur_filter[2][4] +new_image[i+2][j+5] * gauss_blur_filter[2][5] + new_image[i+2][j+6] * gauss_blur_filter[2][6] +new_image[i+3][j] * gauss_blur_filter[3][0] + new_image[i+3][j+1] * gauss_blur_filter[3][1] +new_image[i+3][j+2] * gauss_blur_filter[3][2] + new_image[i+3][j+3] * gauss_blur_filter[3][3] + new_image[i+3][j+4] * gauss_blur_filter[3][4] + new_image[i+3][j+5] * gauss_blur_filter[3][5] + new_image[i+3][j+6] * gauss_blur_filter[3][6]+ new_image[i+4][j] * gauss_blur_filter[4][0] + new_image[i+4][j+1] * gauss_blur_filter[4][1] + new_image[i+4][j+2] * gauss_blur_filter[4][2] + new_image[i+4][j+3] * gauss_blur_filter[4][3] + new_image[i+4][j+4] * gauss_blur_filter[4][4] + new_image[i+4][j+5] * gauss_blur_filter[4][5] + new_image[i+4][j+6] * gauss_blur_filter[4][6] + new_image[i+5][j] * gauss_blur_filter[5][0] + new_image[i+5][j+1] * gauss_blur_filter[5][1] + new_image[i+5][j+2] * gauss_blur_filter[5][2] + new_image[i+5][j+3] * gauss_blur_filter[5][3] + new_image[i+5][j+4] * gauss_blur_filter[5][4] + new_image[i+5][j+5] * gauss_blur_filter[5][5] + new_image[i+5][j+6] * gauss_blur_filter[5][6] + new_image[i+6][j] * gauss_blur_filter[6][0] + new_image[i+6][j+1] * gauss_blur_filter[6][1] + new_image[i+6][j+2] * gauss_blur_filter[6][2] + new_image[i+6][j+3] * gauss_blur_filter[6][3] + new_image[i+6][j+4] * gauss_blur_filter[6][4] + new_image[i+6][j+5] * gauss_blur_filter[6][5] + new_image[i+6][j+6] * gauss_blur_filter[6][6]
    maximum = 0
    for i in range(rows):
      for j in range(columns):
          if maximum < dummy_image_x[i][j] :
              maximum = dummy_image_x[i][j]

    minimum = 0

    for i in range(rows):
      for j in range(columns):
          val = dummy_image_x[i][j]
          constant =(val - minimum) / (maximum - minimum)
          dummy_image_x[i][j] = constant
    return (np.asarray(dummy_image_x))

    def gaussian_blur_matrix(kernlen, nsig):
    #kernlen = 7
    #nsig = 0.707
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

    def difference_of_images(image1,image2):

    rows,columns = image1.shape
    resultant_image = [[0 for x in range(columns)] for y in range(rows)]

    for i in range(rows):
      for j in range(columns):
          resultant_image[i][j] = image1[i][j] - image2[i][j]
    return np.asarray(resultant_image)

    def padding(image):
    rows,columns = len(image),len(image[0])
    pad_image = np.asarray(image)
    new_image = [[0 for x in range(columns + 2)] for y in range(rows + 2 )]
    for i in range(rows+1):
      for j in range(columns+1):
          if i == 0 or i == rows+1:
              new_image[i][j] = 0
          elif j == 0 or j == columns+1:
              new_image[i][j] = 0
          else:
              new_image[i][j] = pad_image[i-1][j-1]
    return new_image

    def key_points_detection_algo(image1,image2,image3):
    imagefirst = np.asarray(image1)
    rows,columns = imagefirst.shape
    image1_new = padding(image1)
    image2_new = padding(image2)
    image3_new = padding(image3)
    key_points_detection = [[0 for x in range(columns)] for y in range(rows)]

    for i in range(rows):
          for j in range(columns):

              value = image2[i][j]
              max_value = max(image1_new[i][j],image1_new[i][j+1],image1_new[i][j+2],image1_new[i+1][j],image1_new[i+1][j+1],image1_new[i+1][j+2],image1_new[i+2][j],image1_new[i+2][j+1],image1_new[i+2][j+2],image2_new[i][j],image2_new[i][j+1],image2_new[i][j+2],image2_new[i+1][j],image2_new[i+1][j+1],image2_new[i+1][j+2],image2_new[i+2][j],image2_new[i+2][j+1],image2_new[i+2][j+2], image3_new[i][j],image3_new[i][j+1],image3_new[i][j+2],image3_new[i+1][j],image3_new[i+1][j+1],image3_new[i+1][j+2],image3_new[i+2][j],image3_new[i+2][j+1],image3_new[i+2][j+2]  )
              min_value = min(image1_new[i][j],image1_new[i][j+1],image1_new[i][j+2],image1_new[i+1][j],image1_new[i+1][j+1],image1_new[i+1][j+2],image1_new[i+2][j],image1_new[i+2][j+1],image1_new[i+2][j+2],image2_new[i][j],image2_new[i][j+1],image2_new[i][j+2],image2_new[i+1][j],image2_new[i+1][j+1],image2_new[i+1][j+2],image2_new[i+2][j],image2_new[i+2][j+1],image2_new[i+2][j+2], image3_new[i][j],image3_new[i][j+1],image3_new[i][j+2],image3_new[i+1][j],image3_new[i+1][j+1],image3_new[i+1][j+2],image3_new[i+2][j],image3_new[i+2][j+1],image3_new[i+2][j+2])
              if value == max_value or value == min_value:
                  key_points_detection[i][j] = 255
              else:
                  key_points_detection[i][j] = 0
    return np.asarray(key_points_detection)

    image = cv2.imread('task2.jpg',0)

    scaling = [1,1/2,1/4,1/8]

    sigma_values = [[0 for x in range(5)] for y in range(4)]

    constant = 1

    for i in range(4):
    for j in range(5):
      if i == 0 and j == 0:
          sigma_values[i][j] = 1 / math.sqrt(2)
          constant = sigma_values[i][j]
          continue
      sigma_values[i][j] = constant * math.sqrt(2)
      constant = sigma_values[i][j]
    constant = sigma_values[i][1]
    #print(constant)

    #print(sigma_values)

    new_image = [[0 for x in range(5)] for y in range(4)]
    for i in range(4):
    for j in range(5):
      print (scaling[i], sigma_values[i][j])
      new_image[i][j] = scalingandblurringoutput(image,scaling[i],scaling[i],7,sigma_values[i][j])
      output_string = "Gaussblur" + str(i) + str(j)+".jpg"
      cv2.imwrite(output_string , new_image[i][j]*255)
    difference_of_gaussians_image  = [[0 for x in range(4)] for y in range(4)]

    for i in range(4):
    for j in range(4):
      difference_of_gaussians_image[i][j] = difference_of_images(new_image[i][j],new_image[i][j+1])
      output_string = "DOG" + str(i) + str(j)+".jpg"
      cv2.imwrite(output_string , difference_of_gaussians_image[i][j]*255)

    key_points_detection = [[0 for x in range(2)] for y in range(4)]
    for i in range(4):
    for j in range(2):
      key_points_detection[i][j] = key_points_detection_algo(difference_of_gaussians_image[i][j],difference_of_gaussians_image[i][j+1],difference_of_gaussians_image[i][j+2])
      output_string = "Keypoints" + str(i) + str(j)+".jpg"
      cv2.imwrite(output_string , key_points_detection[i][j])


    cv2.imwrite('Final_image.jpg',image)
    #cv2.imshow('task2',np.asarray(key_points_detection[0][0]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
