from queue import Empty
import cv2 
import numpy as np
import dlib
import face_recognition
import math

######DRIVER###############
cap=cv2.VideoCapture('v1_driver.mp4')
# cords=np.array([])
cords=[[]]


#########ROAD####################
road=cv2.VideoCapture('v1_road.mp4')

cap.set(cv2.CAP_PROP_POS_FRAMES, 1500)
road.set(cv2.CAP_PROP_POS_FRAMES, 3000)





detector =dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks1.dat")

dist =lambda x1,y1,x2,y2 :(x1-x2)**2-(y1-y2)**2
prevCircle=None


def midpoint(p1,p2):
    return int((p1.x +p2.x)/2),int((p1.y+p2.y)/2)

def maxAndMin(featCoords,mult = 1):
    adj = 10/mult
    listX = []
    listY = []
    for tup in featCoords:
        listX.append(tup[0])
        listY.append(tup[1])
    maxminList = np.array([min(listX)-adj,min(listY)-adj,max(listX)+adj,max(listY)+adj])
    # print(maxminList)
    return (maxminList*mult).astype(int), (np.array([sum(listX)/len(listX)-maxminList[0], sum(listY)/len(listY)-maxminList[1]])*mult).astype(int)
ret_count=0
count=0

while True:
    r_,r_frame=road.read()
    

    _,frame=cap.read()
    
    height,width,_=frame.shape

    r_frame = cv2.resize(r_frame,(int(width*0.5),int(height*0.5)),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    r_x,r_y,_=r_frame.shape
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)



    # feats = face_recognition.face_landmarks(gray)

    # # if len(feats) > 0:
    # leBds, leCenter = maxAndMin(feats[0]['left_eye'], mult=1/0.015)

    # left_eye = frame[leBds[1]:leBds[3], leBds[0]:leBds[2]]
    # # right_eye = frame[reBds[1]:reBds[3], reBds[0]:reBds[2]]

    # # left_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)

    # # left_eye = cv2.resize(left_eye, dsize=(100, 50))

    # # D
    # # isplay the image - DEBUGGING ONLY
    # cv2.imshow('frame', left_eye)

    if bool(faces):
        ret_count=ret_count+1

    font=cv2.FONT_HERSHEY_PLAIN


    
    if  ret_count!= count:
        cv2.putText(frame,"NO LOOKING",(80,200),font,2,(0,0,255),3)

        # print("oppsie"+str(count))
        ret_count=count





    for face in faces:
        # x,y=face.left(),face.top()
        # x1,y1=face.right(),face.bottom()

        # cv2.rectangle(frame,(x,y),(x1,y1),(0,255,0),2)
        landmarks=predictor(gray,face)
    

        left_eye_region=np.array([(landmarks.part(42).x , landmarks.part(42).y),
        (landmarks.part(43).x , landmarks.part(43).y),
        (landmarks.part(44).x , landmarks.part(44).y),
        (landmarks.part(45).x , landmarks.part(45).y),
        (landmarks.part(46).x , landmarks.part(46).y),
        (landmarks.part(47).x , landmarks.part(47).y),

        ],np.int32)
    
        # if bool(landmarks.part(43).x):
        #     ret_count=ret_count+1







        cv2.polylines(frame,[left_eye_region],True,(0,0,255),2)

        height,width,_=frame.shape

        r_height,r_width,_=r_frame.shape

        mask=np.zeros((height,width),np.uint8)
        cv2.polylines(mask,[left_eye_region],True,255,2)
        cv2.fillPoly(mask,[left_eye_region],255)

        left_eye=cv2.bitwise_and(gray,gray,mask=mask)


        min_x=np.min(left_eye_region[:,0])
        max_x=np.max(left_eye_region[:,0])

        min_y=np.min(left_eye_region[:,1])
        max_y=np.max(left_eye_region[:,1])

        gray_eye=left_eye[min_y-5:max_y+5,min_x-5:max_x+5]




        # frame_HSV = cv2.cvtColor(left_eye, cv2.COLOR_BGR2HSV)
        # threshold_eye = cv2.inRange(frame_HSV, (8, 0, 0), (169, 67, 195))

        ######################THRESHOLDING HERE####################################


        gray_eye=cv2.GaussianBlur(gray_eye,(3,3),0)

        # _,threshold_eye=cv2.threshold(gray_eye,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,60,255,cv2.THRESH_BINARY)
        threshold_eye = cv2.adaptiveThreshold(gray_eye,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,103,-25)
        

   #########################################



        height,width=threshold_eye.shape
        left_side_thresh=threshold_eye[0:height,0:int(width/2)]
        left_side_white=cv2.countNonZero(left_side_thresh)


        right_side_thresh=threshold_eye[0:height,int(width/2):width]
        right_side_white=cv2.countNonZero(right_side_thresh)

        # if right_side_white==0:
        if left_side_white ==0:
            gaze_ratio=1
        elif right_side_white==0:
            gaze_ratio=0.2
        else:



          gaze_ratio=left_side_white/right_side_white

    

        cv2.putText(frame,str(gaze_ratio),(50,100),font,2,(0,0,255),3)

        if gaze_ratio<=0.6:
            cv2.putText(frame,"right",(50,200),font,2,(0,0,255),3)


        elif 0.6<gaze_ratio<0.8:
            cv2.putText(frame,"center",(50,200),font,2,(0,0,255),3)


        elif 0.8<=gaze_ratio:
            cv2.putText(frame,"left",(50,200),font,2,(0,0,255),3)






        # cv2.imshow("left",left_side_thresh)
        # cv2.imshow("right",right_side_thresh)







        eye=cv2.resize(gray_eye,None,fx=5,fy=5)
        # cv2.imshow("Eye",eye)


        # if threshold_eye is Empty:



            # print("yo")

             #######################detecting hough circles#############################

        blur=cv2.GaussianBlur(threshold_eye,(3,3),0)

        circles=cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,2,100,param1=100,param2=20,minRadius=17,maxRadius=27) 


           #CHEEK LINE################################################################

        nose_l=(landmarks.part(35).x,landmarks.part(35).y)
        ear_l=(landmarks.part(15).x,landmarks.part(15).y)


        ####DISTANCE OF THE LINE###########
        dist_cheek=(nose_l[0]-ear_l[0])**2+(nose_l[1]-ear_l[1])**2
        per_change=((dist_cheek-150000)/150000)*100

        cv2.putText(frame,"ratios: "+str(dist_cheek/150000),(200,200),font,2,(0,0,255),3)
        cv2.putText(frame,"difference of length: "+str(per_change),(50,400),font,2,(0,0,255),3)
        cv2.putText(frame,"length of the line: "+str(dist_cheek),(50,600),font,2,(0,0,255),3)



        if dist_cheek/150000 >1.17:
            cv2.putText(frame,"RIGHT _L",(50,300),font,2,(0,0,255),3)



        cheek_l=cv2.line(frame,ear_l,nose_l,(0,255,0),2)

        ####################################################

        #################DRAWING RATIO LINES######################################### FOR DRIVER EYE

        no_div=10

        for i in range(no_div):
            div_w=width*i/no_div
            cv2.line(blur, [int((div_w)+div_w*per_change/100),0], [int((div_w)+div_w*per_change/100),height], (255,255,255),5)



        
#######################################################################################




#################################FOR ROAD#########################################

        

        for i in range(no_div):
            r_div_w=r_width*i/no_div
            cv2.line(r_frame, [int(r_div_w+r_div_w*per_change/100),0], [int(r_div_w+r_div_w*per_change/100),r_height], (255,255,255),5)







        ##############################################################################

        if circles is not None:
            circles=np.uint16(np.around(circles))

            chosen=None


            for i in circles[0,:]:
                if chosen is None : chosen=i
                if prevCircle is not None:
                    if dist(chosen[0],chosen[1],prevCircle[0],prevCircle[1]<= dist(i[0],i[1],prevCircle[0],prevCircle[1]  )):
                        chosen=i


            for (x, y, r) in circles[0,:]:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                shape_eye=blur.shape

                # print(shape_eye)

            
                y2=shape_eye[0]/2

                x2=shape_eye[1]

                x1=0
                y1=shape_eye[0]/2


                diff_in_x=x1-x2
                if diff_in_x ==0:
                    diff_in_x=0.000001
                slope=(y1-y2)/(diff_in_x)
                deno=math.sqrt(1+math.pow(slope,2))

                if deno== 0:
                    deno=0.0001



                dist_pup=((y-y1)-(slope)*(x-x1))/(deno)
                cv2.putText(frame,"dispalce: "+str(dist_pup),(50,700),font,2,(0,0,255),3)
                cv2.line(blur, [x1,int(y1)], [x2,int(y2)], (25,25,255),5)
                # print(dist_pup)

                angle = math.atan(0.26*dist_pup/25)

                ratio_test=dist_pup/shape_eye[0]

                # print(str(shape_eye))



                # print(angle/(3.14/2))
                # r_x=road_shape[0]
                # r_y=road_shape[1]
                # print(str(r_x))
                ratio=r_x*6/10

                # ratio_mult=angle/(3.14/2)
                ratio_mult=ratio_test
                
                # cv2.line(r_frame, [0,int(ratio)], [int(r_y),int(ratio)], (25,25,255),5)

                # ratio=r_x*3/4

                

                cv2.line(r_frame, [0,int(ratio+(ratio*ratio_mult))], [int(r_y),int(ratio+(ratio*ratio_mult))], (25,25,255),5)


                



               



                

                for i in range(no_div):
                    if width*i/no_div<=x<width*(i+1)/no_div:
                        rr_div_w=r_width*i/no_div
                        rr_div_w_nex=r_width*(i+1)/no_div
                        # cv2.line(r_frame, [int(rr_div_w+rr_div_w*per_change/100),0], [int(rr_div_w+rr_div_w*per_change/100),r_height], (25,25,255),5)
                        # cv2.line(r_frame, [int(rr_div_w_nex+rr_div_w_nex*per_change/100),0], [int(rr_div_w_nex+rr_div_w_nex*per_change/100),r_height], (25,25,255),5)
                        y_cord_gaze=((rr_div_w+rr_div_w*per_change/100)+(rr_div_w_nex+rr_div_w_nex*per_change/100))/2
                        x_cord_gaze=ratio+(ratio*ratio_mult)

                        cords.append((y_cord_gaze,x_cord_gaze))

                        cv2.circle(r_frame,[int(y_cord_gaze),int(x_cord_gaze)] , 50, [0,255,0], 5)


                        # cv2.line(r_frame, [int(r_width*i/no_div),0], [int(r_width*i/no_div),r_height], (25,25,225),5)
                        # cv2.line(r_frame, [int(r_width*(i+1)/no_div),0], [int(r_width*(i+1)/no_div),r_height], (25,25,255),5)



                # if width/10<=x<width/5:
                #     print("1")
                #     cv2.line(r_frame, [int(r_width/10),0], [int(r_width/10),r_height], (25,25,225),5)
                #     cv2.line(r_frame, [int(r_width/5),0], [int(r_width/5),r_height], (25,25,255),5)


                #     # cv2.rectangle(frame,(x,y),(x1,y1),(0,255,0),2)
                
                # elif width/5<=x<width*3/10:
                #     cv2.line(r_frame, [int(r_width/5),0], [int(r_width/5),r_height], (25,25,255),5)
                #     cv2.line(r_frame, [int(r_width*3/10),0], [int(r_width*3/10),r_height], (25,25,255),5)
                        
                #     print("2")
                # elif width*3/10<=x<width*2/5:

                #     cv2.line(r_frame, [int(r_width*3/10),0], [int(r_width*3/10),r_height], (25,25,255),5)
                #     cv2.line(r_frame, [int(r_width*2/5),0], [int(r_width*2/5),r_height], (25,25,255),5)


                #     print("3")

                # elif width*2/5<=x<width/2:

                #     cv2.line(r_frame, [int(r_width*2/5),0], [int(r_width*2/5),r_height], (25,25,255),5)
                #     cv2.line(r_frame, [int(r_width/2),0], [int(r_width/2),r_height], (25,25,255),5)


                #     print("4")

                # elif width/2<=x<width*6/10:
                #     cv2.line(r_frame, [int(r_width/2),0], [int(r_width/2),r_height], (25,25,255),5)
                #     cv2.line(r_frame, [int(r_width*6/10),0], [int(r_width*6/10),r_height], (25,25,255),5)


                #     print("5")
                # elif width*6/10<=x<width*7/10:
                #     cv2.line(r_frame, [int(r_width*6/10),0], [int(r_width*6/10),r_height], (25,25,255),5)
                #     cv2.line(r_frame, [int(r_width*7/10),0], [int(r_width*7/10),r_height], (25,25,255),5)


                #     print("6")

                # elif width*7/10<=x<width*8/10:
                #     cv2.line(r_frame, [int(r_width*7/10),0], [int(r_width*7/10),r_height], (25,25,255),5)
                #     cv2.line(r_frame, [int(r_width*8/10),0], [int(r_width*8/10),r_height], (25,25,255),5)


                #     print("7")
                # elif width*8/10<=x<width*9/10:
                #     cv2.line(r_frame, [int(r_width*8/10),0], [int(r_width*8/10),r_height], (25,25,255),5)
                #     cv2.line(r_frame, [int(r_width*9/10),0], [int(r_width*9/10),r_height], (25,25,255),5)


                #     print("8")







                




            cv2.circle(blur
            ,(chosen[0],chosen[1]),1,(0,100,100),3)
            cv2.circle(blur
            ,(chosen[0],chosen[1]),chosen[2],(255,0,255),3)
            prevCircle=chosen

            sma_val=10


            if count<sma_val:
                cv2.circle(r_frame,[int(y_cord_gaze),int(x_cord_gaze)] , 50, [0,255,0], 5)

            else:
                # print(cords[count])

                
                avg_y=0
                avg_x=0
                for i in range(sma_val):
                    print("c- " +str(count))
                    print("dis- "+str(count-i))

                    avg_y=avg_y+cords[count][0]
                    avg_x=avg_x+cords[count][1]
                avg_y=avg_y/sma_val
                avg_x=avg_x/sma_val
            
                cv2.circle(r_frame,[int(avg_y),int(avg_x)] , 50, [0,255,0], 5)

##################################################################################################3

##################verticle coord reference lines##########################
        lip_b=(landmarks.part(8).x,landmarks.part(58).y)

        chin=(landmarks.part(8).x,landmarks.part(8).y)






#####################################################################3

#############IM SHOW ##############################################3
        cv2.imshow("threshold",threshold_eye)
        cv2.imshow("blur",blur)
        # cv2.imshow("mask",left_eye)



        

        
        # x=landmarks.part(46).x
        # y=landmarks.part(46).y
        # cv2.circle(frame,(x,y),3,(0,0,255),2)

        left_point=(landmarks.part(42).x,landmarks.part(42).y)
        right_point=(landmarks.part(45).x,landmarks.part(45).y)

        #3#################################################3
     


        

        center_top=midpoint(landmarks.part(43),landmarks.part(44))
        center_bottom=midpoint(landmarks.part(47),landmarks.part(46))
        

        hor_line=cv2.line(frame,left_point,right_point,(0,255,0),2)
        ver_line=cv2.line(frame,center_bottom,center_top,(0,255,0),2)
        hor_line=cv2.line(frame,lip_b,chin,(0,255,0),2)
        


        

        landmarks=None






#58 9
#34 52


    fps = cap.get(cv2.CAP_PROP_FPS)
    r_fps = road.get(cv2.CAP_PROP_FPS)
    # print("frame : "+str(fps))
    # print("r_frame : "+str(r_fps))

    cv2.imshow("Frame",frame)
    cv2.imshow("road",r_frame)

    count=count+1

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
road.release()
cv2.destroyAllWindows()