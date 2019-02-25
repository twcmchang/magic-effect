import numpy as np

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