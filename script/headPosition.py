#! /usr/bin/env python3.8

import cv2
from matplotlib.pyplot import axes
import mediapipe as mp
import numpy as np
import time
import sys
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float64MultiArray, MultiArrayDimension
from math import *


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2,refine_landmarks=True,min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)    
mp_drawing_styles = mp.solutions.drawing_styles



class mediapipe_face: 
    def __init__(self):
        self.bridge = CvBridge()
        self.face_pub = rospy.Publisher("/face",Float64MultiArray,queue_size=10)
        self.image_sub = rospy.Subscriber("/image/image_raw",Image,self.callback)
        print("Ready..")


    def callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    #         # depth
    #         #image = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height, data.width, -1)
    #         #print(image.min(), image.max())
    #         #image = cv2.convertScaleAbs(image, 255/image.max())
    #         #print(image.min(), image.max())

        except CvBridgeError as e:
            print(e)
        start = time.time()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
        image.flags.writeable = False
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    # Get the result
        results = face_mesh.process(image)
    
    # To improve performance
        image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        # definir le model 3D
        face_3d = np.array([[0.000000* img_w, -3.406404*img_h, 5.979507* 30],
                            [-2.266659* img_w, -7.425768*img_h, 4.389812* 30],
                            [-0.729766* img_w, -1.593712*img_h, 5.833208* 30],
                            [-1.246815* img_w, 0.230297*img_h, 5.681036* 30],
                            [2.266659* img_w, -7.425768*img_h, 4.389812* 30],
                            [0.729766* img_w, -1.593712*img_h, 5.833208* 30]
                            ])
        #face_2d = np.empty((1,12),np.float64)
        
        face = np.empty((0,15),np.float64)
        

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_2d = []
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                        #np.append(face_2d[face_id],[x, y])
                        face_2d.append([x,y])

                face_2d = np.array(face_2d, dtype=np.float64)
                #print(face_2d)
                
            # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

            # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                

            # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

            # # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)


            # Get the y rotation degree
                x = angles[0] 
                y = angles[1] 
                z = angles[2] 

            # regroupe face data [[pitch,yaw,roll,x1,y1,x2,y2,....]]
                face =np.append(face,[np.append(angles,face_2d)],axis=0)
                
            #Add the text on the image
                cv2.putText(image, "pitch: " + str(np.round(x,2)), (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "yaw: " + str(np.round(y,2)), (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "roll: " + str(np.round(z,2)), (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec)

            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime
            cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)


            #Publish face data
            face_msg = Float64MultiArray()
            face_msg.data = np.ravel(face).tolist()
            face_msg.layout.data_offset = 0
            face_msg.layout.dim = [ MultiArrayDimension("faces",len(face.tolist()),15*len(face.tolist())),
                                    MultiArrayDimension("features",15,1)]
            
            self.face_pub.publish(face_msg)
        cv2.imshow('Head Pose Estimation', image)
        cv2.waitKey(1)



def main(args):
    rospy.init_node('mediapipe_face', anonymous=True)
    mp_face = mediapipe_face()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
