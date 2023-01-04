import numpy as np 
import cv2
def z_axis_rotation_matrix(ang):
    RT = np.eye(4, 4)
    RT[0,0] = +cos(ang)
    RT[0,1] = -sin(ang)
    RT[1,0] = +sin(ang)
    RT[1,1] = +cos(ang)
    return RT

def y_axis_rotation_matrix(ang):
    RT = np.eye(4, 4)
    RT[0,0] = +cos(ang)
    RT[0,2] = sin(ang)
    RT[2,0] = -sin(ang)
    RT[2,2] = +cos(ang)
    return RT
def x_axis_rotation_matrix(ang):
    RT = np.eye(4, 4)
    RT[1,1] = +cos(ang)
    RT[1,2] = -sin(ang)
    RT[2,1] = sin(ang)
    RT[2,2] = +cos(ang)
    return RT

def draw_img_points(img, img_pts, color):
    for in in range(len(img_pts)):
        cv2.circle(img, img_pts[i], 1, color, CV_FILLED)