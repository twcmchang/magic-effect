import numpy as np
import cv2
from model_deeplab import *

COLORS = [(252,141,98),(102,194,165), (141,160,203),(231,138,195),(166,216,84)]

####################
#     Functions    #
####################

def run_segmentation(f):
  """Inferences DeepLab model and visualizes result."""
  try:
      if isinstance(f, str):
          original_img = Image.open(f)
      elif isinstance(f, np.ndarray):
          original_img = Image.fromarray(f)
  except:
      print("Unexpected error: invalid input")
      raise
  rgb_img, seg_map = MODEL.run(original_img)
  return rgb_img, seg_map

def find_cntr(cv2_img, n_max=5, th=1000):
  '''input: cv2_img = cv2.imread(img_path)'''
  '''output: mask '''
  b,g,r = cv2.split(cv2_img)  
  img = cv2.merge([r,g,b])
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  _,th2 = cv2.threshold(gray,10,1,cv2.THRESH_BINARY)
  contours, hierarchy = cv2.findContours(th2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
  areas = [-cv2.contourArea(contour) for contour in contours]
  num = max(n_max, np.sum(np.array(areas) < (-1)*th))
  return [contours[top] for top in np.argsort(areas)[:num]]

def get_outline(contour, color, width=10):
  element = cv2.getStructuringElement(cv2.MORPH_RECT,(width,width))
  dilate = cv2.dilate(contour, element)
  outline = cv2.absdiff(dilate, contour)
  retval, outline = cv2.threshold(outline, 0, 255, cv2.THRESH_BINARY)
  colored_outline = np.stack((outline / 255.0 * color[0],
                              outline / 255.0 * color[1],
                              outline / 255.0 * color[2]), axis=2).astype(np.uint8)
  outline_image = cv2.medianBlur(colored_outline, 1) # to make it smoother
  return outline_image

def get_offset(contour, color, offset_x, offset_y):
  colored_contour = np.stack((contour / 255.0 * color[0],
                              contour / 255.0 * color[1],
                              contour / 255.0 * color[2]), axis=2).astype(np.uint8)
  num_rows, num_cols = colored_contour.shape[:2]
  translation_matrix = np.float32([ [1, 0, offset_x], [0, 1, offset_y] ])
  offset_image = cv2.warpAffine(colored_contour, translation_matrix, (num_cols, num_rows))
  
  return offset_image

####################
#      Effects     #
####################

def outline_effect(f):
  rgb, fore = run_segmentation(f)
  
  # find contours on person semantic mask
  contours = find_cntr(fore)
  
  # select at most five contours
  num = min(len(contours), 5)
  
  # combine all contours with different colors
  canvas = np.zeros(rgb.shape, np.uint8)
  for i in range(num):
      contour = cv2.drawContours(np.zeros(rgb.shape[0:2], np.uint8), contours, i, (255,255,255), -1)
      canvas += get_outline(contour, color=COLORS[i], width=10)
  
  # combine with original images
  rgb_new = rgb*(1-(canvas>0)) + canvas
  return rgb, rgb_new 

def offset_effect(f):
  rgb, fore = run_segmentation(f)

  # find contours on person semantic mask
  contours = find_cntr(fore)

  # select at most five contours
  num = min(len(contours), 5)

  # combine all contours with different colors
  canvas = np.zeros(rgb.shape, np.uint8)
  for i in range(num):
      contour = cv2.drawContours(np.zeros(rgb.shape[0:2], np.uint8), contours, i, (255,255,255),-1)
      canvas += get_offset(contour, color=COLORS[i], offset_x=10, offset_y=10)
  
  # to avoid overlapping on person cutouts
  canvas = canvas * (1-(fore>0))

  # combine with original images
  rgb_new = rgb*(1-(canvas>0)) + canvas
  return rgb, rgb_new

def rotate_vector(vector, degree):
    """
    @vector: two dimension input like [1, 0], (1,0), or np.array([1, 0])
    @degree: rotation angle
    
    [ cos0 -sin0
      sin0  cos0 ]
    
    return roration vector
    """
    cos = np.cos(np.pi/180 * degree)
    sin = np.sin(np.pi/180 * degree)
    
    return [vector[0] * cos - vector[1] * sin, vector[0] * sin + vector[1] * cos]

def get_extrapolated_contour(contour_points, extrapolate_length, margin_to_contour, extrapolate_angle):
    """
    add a specific pattern between any two consecutive points (a, b)

    Arguments
    @points: points of interest
    @length: the length of the line
    @space:  the space between the contour and perpendicular lines
    @angle:  the rotation angle of the vector, b - a
    
    Return
    - np.array(p1): the start point of lines
    - np.array(p2): the end point of lines
    """
    p1 = list()
    p2 = list()
    for i in range(len(contour_points)-1):
        a = contour_points[i]
        b = contour_points[i+1]
        midx, midy = (a+b)/2
        
        vx, vy = b - a
        mag = np.sqrt( vx**2 + vy**2)
        
        vx = vx/mag
        vy = vy/mag
        
        wx, wy = rotate_vector(vector=[vx, vy], degree=extrapolate_angle)
        
        cx = int(midx + margin_to_contour*wx)
        cy = int(midy + margin_to_contour*wy)
        
        dx = int(cx + wx * extrapolate_length)
        dy = int(cy + wy * extrapolate_length)
        
        p1.append([cx, cy])
        p2.append([dx, dy])
    return np.array(p1), np.array(p2)