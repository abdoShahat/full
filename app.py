from flask import Flask,render_template,Response,request
import cv2
from PIL import Image
from imutils import face_utils
import numpy as np 
from utils import *
# from flask_mobility import Mobility


app = Flask(__name__)
camera = cv2.VideoCapture(0)
i = 1
color=[170,185,170]
# color=(170,185,170)
def generat_frame():
    
    while True:

        # read the camera frame 
        success,frame=camera.read()
        if not success:
            break
        else:
            
            frame = cv2.flip(frame,1)
            feat_allied = apply_makeup(frame, True, 'shadow', False)

            # # Landmark model location
            # predictor_path = "dlib/shape_predictor_68_face_landmarks.dat"

            # # Get the face detector
            # faceDetector = dlib.get_frontal_face_detector()
            
            # # The landmark detector is implemented in the shape_predictor class
            # landmarkDetector = dlib.shape_predictor(predictor_path)

            # # imDlib = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            # imDlib = frame
            # # Change this color to change lipstick color!
            # # color = colors

            # # Change the Choice of image here
            # choice = imDlib
            # imcopy = choice.copy()

            # points = fbc.getLandmarks(faceDetector, landmarkDetector, choice)

            # # Get points of the lips

            # # I will be using cv2.pollyFill to build the mask, polyfill takes in vectorized set of points
            # # this means a line will be drawn between point[0] -> point[1] -> point[2] -> ... -> point[N] -> point[0] and subsequently filled

            # # So this means how upperlips and lowerlips are ordered MATTERS or else the mask will be drawn incorrectly
            # upperlips = points[43:48]
            # lowerlips = points[37:42] 

            # # Version 2 of lip mask

            # # cv2.pollyFill wants np.arrays to be passed to it. Currently upperlips and lowerlips are a list(tuples)

            # # They need to be converted from list(tuples) to list(list(int))
            # uHull = [[p[0],p[1]] for p in upperlips]
            # # for p in upperlips:
            # #   uHull.append([p[0], p[1]])
            # lHull = [[p[0],p[1]] for p in lowerlips]
            # # for p in lowerlips:
            # #   lHull.append([p[0], p[1]])
            # uHull = np.array(uHull)
            # lHull = np.array(lHull)

            # # We build the mask for the lips
            # row, col, _ = choice.shape
            # mask = np.zeros((row, col), dtype=choice.dtype)

            # cv2.fillPoly(mask, [uHull], (255));
            # cv2.fillPoly(mask, [lHull], (255));

            # bit_mask = mask.astype(np.bool)

            # # Find bounding box for mask preview
            # lst = upperlips + lowerlips
            # xmin, xmax = min(lst, key = lambda i : i[1])[1], max(lst, key = lambda i : i[1])[1]
            # ymin, ymax = min(lst, key = lambda i : i[0])[0], max(lst, key = lambda i : i[0])[0]

            # pixel = np.zeros((1,1,3), dtype=np.uint8)
            # r_ = 0
            # g_ = 1
            # b_ = 2

            # pixel[:,:,r_], pixel[:,:,g_], pixel[:,:,b_] = color[r_], color[g_], color[b_]

            # out = choice.copy()

            # # Convert image of person from RGB to HLS
            # pixel_hsl = cv2.cvtColor(pixel, cv2.COLOR_RGB2HLS)
            # outhsv = cv2.cvtColor(out,cv2.COLOR_RGB2HLS)
            # channel = 0
            # # st.write('color in def is :',color)
            # # extract the hue channels
            # hue_img = outhsv[:,:,channel]
            # hue_pixel = pixel_hsl[:,:,0]

            # hue_img[bit_mask] = hue_pixel[0,0]

            # out = cv2.cvtColor(outhsv,cv2.COLOR_HLS2RGB)

            
            ret,buffer=cv2.imencode('.jpg',feat_allied)
            frame=buffer.tobytes()
        
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generat_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=="__main__":
    app.run(debug=True)

