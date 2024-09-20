from imutils import face_utils
import numpy as np
import dlib
import cv2
import time
import argparse
import shutil
from utils.utils import *
from utils.dataset import *

class FaceEyeDetectionDlib:
    def __init__(self, model="shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model)
        print("[INFO] Face Landmark Detection Model loaded....")

        self.left_eye_index = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self.right_eye_index = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    @staticmethod
    def _process_eye_box_(box):
        box = list(box)
        box[1] -= int(box[3]*4)
        box[3] += int(box[3]*3)*2

        box[0] -= int(box[2]*0.6)
        box[2] += int(box[2]*0.6)*2
        return box

    @staticmethod
    def _preprocess_face_box_(rect):
        box = face_utils.rect_to_bb(rect)
        box = list(box)
        box[1] -= int(box[3]*4)
        box[3] += int(box[3]*4)
        return box

    def detect_faces(self, image):
        rects = []
        if image is not None:
            gray = image
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale frame
            rects = self.detector(gray, 0)
            bbs = [FaceEyeDetectionDlib._preprocess_face_box_(rect) for rect in rects]
        return rects, bbs

    def detect_eyes(self, image, face_rects):
        face_landmarks = []
        if image is not None:
            gray = image
            if len(image.shape) > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for rect in face_rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            left_eye_box = cv2.boundingRect(np.array([shape[self.left_eye_index[0]:self.left_eye_index[1]]]))
            left_eye_box = FaceEyeDetectionDlib._process_eye_box_(left_eye_box)
            right_eye_box = cv2.boundingRect(np.array([shape[self.right_eye_index[0]:self.right_eye_index[1]]]))
            right_eye_box = FaceEyeDetectionDlib._process_eye_box_(right_eye_box)

            face_landmarks.append((FaceEyeDetectionDlib._preprocess_face_box_(rect),left_eye_box, right_eye_box))
        return face_landmarks
    
    def extract_faces(self, image, shapes):
        images = []
        for face in shapes:
            x,y,w,h = face
            face = image[y:y+h, x:x+w]
            images.append(face)
        return images

    def extract_faces_eye(self, image, shapes, extract_face=False):
        images = []
        for faceBB, leyeBB, reyeBB in shapes:
            if extract_face:
                x,y,w,h = faceBB
                face = image[y:y+h, x:x+w]
                x,y,w,h = leyeBB
                leye = image[y:y+h, x:x+w]
                x,y,w,h = reyeBB
                reye = image[y:y+h, x:x+w]
                images.append((face, leye, reye))
            else:
                x,y,w,h = leyeBB
                leye = image[y:y+h, x:x+w]
                x,y,w,h = reyeBB
                reye = image[y:y+h, x:x+w]
                images.append((leye, reye))
        return images

    def draw_faces_eyes(self, image, boxes):
            """
            Accepts single image and detected face boxes and returns image with rectangles drawn on face locations.
            image : single image
            boxes : x,y,w,h of face detected i.e face coordinates
            """
            image_cp = None
            try:
                image_cp = image.copy()
                # print(boxes)
                for (faceBB, leyeBB, reyeBB) in boxes:
                    # print(1)
                    (x,y,w,h) = faceBB
                    cv2.rectangle(image_cp, (x,y), (x+w, y+h), (0,0,255), int(0.01*image_cp.shape[0]))
                    (x,y,w,h) = leyeBB
                    cv2.rectangle(image_cp, (x,y), (x+w, y+h), (0,255,255), int(0.01*image_cp.shape[0]))
                    (x,y,w,h) = reyeBB
                    cv2.rectangle(image_cp, (x,y), (x+w, y+h), (255,0,255), int(0.01*image_cp.shape[0]))

            except Exception as e:
                print(f"[ERROR] Draw {e}")
                
            return image_cp


if __name__ == "__main__":
    from os.path import join
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', dest='dataset', type=str, default='CasiaIris', help='Main folder of the dataset')
    arguments = parser.parse_args()
    datasetMainFolder = r'D:\8m\Etude\Fac centrale\M2 ISII\S4\meet2\Nouveau dossier\wiam\datasets'

    datasetFolder = datasetMainFolder + arguments.dataset + "/"
    
      	# Get Dataset information
    classes, filenames, xFilepaths, y = getDataset(arguments.dataset)
    print("Launching detection on the " + arguments.dataset + " dataset...")
    startTime = time.time()
    
      	# This counter is used to store the png 
    fd = FaceEyeDetectionDlib(join("models","shape_predictor_68_face_landmarks.dat"))
    Dmain=r'D:\8m\Etude\Fac centrale\M2 ISII\S4\meet2\Nouveau dossier\wiam'
    os.chdir(Dmain) 
    if not os.path.exists('CASIA-IRIS-Face'):
        os.mkdir('CASIA-IRIS-Face')
    else:
        shutil.rmtree('CASIA-IRIS-Face')
        os.mkdir('CASIA-IRIS-Face')
    os.chdir(Dmain) 
    if not os.path.exists('CASIA-IRIS-Left-Eye'):
        os.mkdir('CASIA-IRIS-Left-Eye')
    else:
        shutil.rmtree('CASIA-IRIS-Left-Eye')
        os.mkdir('CASIA-IRIS-Left-Eye')
    os.chdir(Dmain) 
    if not os.path.exists('CASIA-IRIS-Right-Eye'):
        os.mkdir('CASIA-IRIS-Right-Eye')
    else:
        shutil.rmtree('CASIA-IRIS-Right-Eye')
        os.mkdir('CASIA-IRIS-Right-Eye')
    DF = r'D:\8m\Etude\Fac centrale\M2 ISII\S4\meet2\Nouveau dossier\wiam\CASIA-IRIS-Face'
    DLE = r'D:\8m\Etude\Fac centrale\M2 ISII\S4\meet2\Nouveau dossier\wiam\CASIA-IRIS-Left-Eye'
    DRE = r'D:\8m\Etude\Fac centrale\M2 ISII\S4\meet2\Nouveau dossier\wiam\CASIA-IRIS-Right-Eye'
    os.chdir(Dmain)
    counter = 1
    NumAC = 0
    NumC = 0
    allpho = 0
    pho = 0
    for xfp in xFilepaths: 
        classe = xfp[0]+xfp[1]
        NumC = int(classe)
        if (NumC>NumAC):
            #counter = 1
            NumAC = NumAC + 1
            os.chdir(DF) 
            if not os.path.exists(str(NumC)):
                os.mkdir(str(NumC))
            else:
                shutil.rmtree(str(NumC))
                os.mkdir(str(NumC))
            os.chdir(DLE) 
            if not os.path.exists(str(NumC)):
                os.mkdir(str(NumC))
            else:
                shutil.rmtree(str(NumC))
                os.mkdir(str(NumC))
            os.chdir(DRE) 
            if not os.path.exists(str(NumC)):
                os.mkdir(str(NumC))
            else:
                shutil.rmtree(str(NumC))
                os.mkdir(str(NumC))
        os.chdir(Dmain)
        frame = imgRead(datasetFolder + xfp )
        allpho = allpho + 1
        try:
                    rects, boxes = fd.detect_faces(frame)
                    boxes = fd.detect_eyes(frame, rects)
                    image_cp = None
                    image_cp = frame.copy()
                    for faceBB, leyeBB, reyeBB in boxes:
                            pho = pho + 1
                            (x,y,w,h) = faceBB
                            cv2.rectangle(image_cp, (x,y), (x+w, y+h), (0,0,255), int(0.01*image_cp.shape[0]))
                            face = image_cp[y:y+h, x:x+w]
                            DFS = DF + "\\" + str(NumC)
                            os.chdir(DFS)
                            cv2.imwrite('Face'+str(counter)+'.jpg', face)
                            os.chdir(Dmain) 
                            (x,y,w,h) = leyeBB
                            cv2.rectangle(image_cp, (x,y), (x+w, y+h), (0,255,255), int(0.01*image_cp.shape[0])) 
                            leye = image_cp[abs(y):abs(y+h), abs(x):abs(x+w)]
                            DLES = DLE + "\\" + str(NumC)
                            os.chdir(DLES) 
                            cv2.imwrite('leftEye'+str(counter)+'.jpg', leye)
                            os.chdir(Dmain) 
                            (x,y,w,h) = reyeBB
                            cv2.rectangle(image_cp, (x,y), (x+w, y+h), (255,0,255), int(0.01*image_cp.shape[0]))
                            reye = image_cp[abs(y):abs(y+h), abs(x):abs(x+w)]
                            DRES = DRE + "\\" + str(NumC)
                            os.chdir(DRES) 
                            cv2.imwrite('RightEye'+str(counter)+'.jpg', reye)
                            os.chdir(Dmain) 
                    #cv2.imwrite('AllDraw.jpg',image_cp)
                    if frame is None:
                        exit(1)
        except Exception as e:
            print(f"[ ERROR ] : {e}")
        finally:
            cv2.destroyAllWindows() 
        counter = counter + 1
    print("--- Détection effectuée en %.2f secondes ---" % (time.time() - startTime))
    print("--- %d images détectées sur %d images  ---" % (pho , allpho))
    pourcentage = (pho*100)/allpho
    print(f"--- Taux de détection = {pourcentage:.2f} %---")