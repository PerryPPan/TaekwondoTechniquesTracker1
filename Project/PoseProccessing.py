import cv2
import mediapipe as mp
import numpy as np
import sys
import os
import random
from sklearn.cluster import KMeans


def initialize_model():
    mp_drawing = mp.solutions.drawing_utils  # creates the class to draw landmarks on the pose
    mp_drawing_styles = mp.solutions.drawing_styles  # creates style that we use to draw
    mp_pose = mp.solutions.pose  # IMPORTANT: initializes the pose object, this is what we use to create the poses on the picture/video
    return mp_pose, mp_drawing


def make_directory(name: str):
    # checks if there already is this directory, if there isn't then we need to create it
    if not os.path.isdir(name):
        os.mkdir(name)


def initialize_video_writer(imagename, videopath):
    # if os.path is used then it automatically splits the video path into two parts head and tail
    # os.path.basename returns the tail (or the last part of the path)
    basename = os.path.basename(videopath)
    # this splits it into a root and an extension the root is what comes before the dot and the extension is what comes after and with the dot
    filename, extension = os.path.splitext(basename)
    size = (480, 640)
    # it calls the image name in the function
    make_directory(imagename)
    out = cv2.VideoWriter(f"{imagename}/{filename}_out.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, size)
    print(f"{imagename}/{filename}_out.avi")
    return out


def resize_image(image):
    # imports the image height and width plus one more which is substituted by an underscore
    h, w, _ = image.shape
    # it makes a new height and width by square rooting both
    h, w = h // 2, w // 2
    # this resizes the image into the new width and height that we just set
    image = cv2.resize(image, (w, h))
    return image, h, w


def pose_process_image(image, pose):
    # first two lines improves the performance which helps the processing faster
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # processed image, not drawn yet, create the poses for each video frame
    results = pose.process(image)

    # reverts because we are done processing
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def plot_angles_from_frames(mp_pose, landmarks, image, h, w):
    angles = []
    val = 50
    angle, image = plot_angle(mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                              mp_pose.PoseLandmark.LEFT_ELBOW.value,
                              mp_pose.PoseLandmark.LEFT_WRIST.value, landmarks, image, h, w + val)
    angles.append(angle)
    angle, image = plot_angle(mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                              mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                              mp_pose.PoseLandmark.RIGHT_WRIST.value, landmarks, image, h, w - val)
    angles.append(angle)
    angle, image = plot_angle(mp_pose.PoseLandmark.LEFT_HIP.value,
                              mp_pose.PoseLandmark.LEFT_KNEE.value,
                              mp_pose.PoseLandmark.LEFT_ANKLE.value, landmarks, image, h, w + val)
    angles.append(angle)
    angle, image = plot_angle(mp_pose.PoseLandmark.RIGHT_HIP.value,
                              mp_pose.PoseLandmark.RIGHT_KNEE.value,
                              mp_pose.PoseLandmark.RIGHT_ANKLE.value, landmarks, image, h, w - val)
    angles.append(angle)
    angle, image = plot_angle(mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                              mp_pose.PoseLandmark.LEFT_HIP.value,
                              mp_pose.PoseLandmark.LEFT_KNEE.value, landmarks, image, h, w + val)
    angles.append(angle)
    angle, image = plot_angle(mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                              mp_pose.PoseLandmark.RIGHT_HIP.value,
                              mp_pose.PoseLandmark.RIGHT_KNEE.value, landmarks, image, h, w - val)
    angles.append(angle)
    angle_wrist_shoulder_hip_left, image = plot_angle(mp_pose.PoseLandmark.LEFT_WRIST.value,
                                                      mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                                                      mp_pose.PoseLandmark.LEFT_HIP.value, landmarks, image, h, w + val)
    angles.append(angle_wrist_shoulder_hip_left)
    angle_wrist_shoulder_hip_right, image = plot_angle(mp_pose.PoseLandmark.RIGHT_WRIST.value,
                                                       mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                                                       mp_pose.PoseLandmark.RIGHT_HIP.value, landmarks, image, h,
                                                       w - val)
    angles.append(angle_wrist_shoulder_hip_right)

    return angles  # returning the angles for the joints of a single video frame


def calculate_angle(a, b, c):
    a = np.array(a)  # first point for the x and y coordinates since the function is called in plot angle
    b = np.array(b)  # middle point for the x and y coordinates since the function is called in plot angle
    c = np.array(c)  # end point for the x and y coordinates since the function is called in plot angle

    # calculates the angle using trigonometry
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    # if the angle is bigger than 180 degrees then it turns it into a smaller angle
    if angle > 180.0:
        angle = 360 - angle
    return round(angle, 1)


def plot_angle(p1, p2, p3, landmarks, image, h, w):
    # get coordinates
    a = [landmarks[p1].x, landmarks[p1].y]  # this sets the list to the x and y coordinates to the first point
    b = [landmarks[p2].x, landmarks[p2].y]  # this sets the list to the x and y coordinates to the second point
    c = [landmarks[p3].x, landmarks[p3].y]  # this sets the list to the x and y coordinates to the third point

    # this calls the calculate angle function which calculates the angle of the coordinates
    angle = calculate_angle(a, b, c)
    # this draws the angles
    draw_angle(tuple(np.multiply(b, [w, h]).astype(int)), image, angle)
    return angle, image  # angle is the angle between a b and c, image is the image with the landmarks drawn on it


def draw_angle(org: tuple, image, angle):
    # this sets the font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # sets the font size
    fontScale = 0.4
    # sets the color of the font
    color = (255, 255, 255)
    # sets the thickness of the font
    thickness = 1
    # puts the texts onto the image
    image = cv2.putText(image, str(angle), org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    return image


# the results is the processed image with the pose the mp_drawing is what we use to draw the pose is what we use to set the pose, the image is the image

def draw_landmarks(results, mp_drawing, mp_pose, image):
    # this makes all the landmarks listed in the list invisible since we don't use them
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        if idx in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 29, 30, 31, 32]:
            results.pose_landmarks.landmark[idx].visibility = 0
    print('here')
    # this connects the pose by drawing the landmarks and taking the processed image
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
    return image  # it returns the image with the landmarks and connections


def get_frames_angles(imagename: str, videopath: str) -> tuple:
    mp_pose, mp_drawing = initialize_model()  # this creates the class to draw landmarks on the pose and
    # the style we use to draw, calls the function initialize_model, which creates the class for
    # pose processing and another class for the landmark drawing
    cap = cv2.VideoCapture(videopath)  # takes a video path and stores it into a usable video object in python
    out = initialize_video_writer(imagename, videopath)
    img_count = 0  # it keeps tracks of the image, whenever a new image gets added into code it goes up by 1
    output_images = []  # iamges that are put out
    frames = []  # keeps track of all the angles for each frame of the video, the angle list is going to be appended into this list

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:  # sets all of these as pose and initializes it
        while cap.isOpened():  # while the video is opened/being used
            success, image = cap.read()  # takes in a video path and stores it into a usable video object in python

            if not success:  # if the program didn't succeed in running then it will print ignoring empty camera frame and break
                print("ignoring empty camera frame.")
                break
            image, h, w = resize_image(image)  # resizes the image

            image, results = pose_process_image(image, pose)  # helps with processing the code faster and better
            # image is the video frame, and results are the pose data waiting to be drawn

            try:
                landmarks = results.pose_landmarks.landmark  # sets landmarks as the pose data waiting to be drawn
                angles = plot_angles_from_frames(mp_pose, landmarks, image, h,
                                                 w)  # sets angles as the angles for the joints of a single video frame
                frames.append(
                    angles)  # appends angles for the joints of a single video frame into the list frames which keeps track of all the angles for each frame of the video
                # cv2.imshow('image', image)
                image = draw_landmarks(results, mp_drawing, mp_pose, image)  # draws the angles
                out.write(image)  # outwrites the image
                # cv2.imshow('display', image) # in python IDE, change cv2_imshow to cv2.imshow('title of frame/image', image)

                outImageFile = f"{imagename}/{imagename}{img_count}.jpg"
                print(outImageFile)
                cv2.imwrite(outImageFile, image)  # writes the image
                img_count += 1  # adds one to the variable img_count
            except:  # if the above doesn't work then it moves on to the next part of the code
                pass

            if cv2.waitKey(5) & 0xFF == 27:  # if it presses a specfic key then it quits
                break

    cap.release()
    out.release()

    return frames  # returns the list with all the data of the angles

def get_nearest_neighbor(image, indexes, frames):
    a = np.array(image)
    min_dist = sys.maxsize
    nearest = indexes[0]
    for idx in indexes:
        b = np.array(frames[idx])
        dist = np.linalg.norm(a - b)
        if min_dist > dist:
            nearest = idx
            min_dist = dist
    return nearest

def main(video1, video2):
    coach_frames = get_frames_angles(imagename='coach',
                                     videopath=video1)
    student_frames = get_frames_angles(imagename='student',
                                       videopath=video2)

    for frame in student_frames:
        print(frame)
    print(len(student_frames))
    print(student_frames, "dkkdfuh")

    for frame in coach_frames:  # for each data for an angle it will do what is listed at the bottom
        print(frame)  # print the angle data for this specific angle in the list
    print(len(coach_frames))  # prints how many elements are in coach_frames

    student_n_cluster = 16  # Variable for Kmeans to see how many clusters to make
    print(student_n_cluster)
    X = np.array(student_frames)
    kmeans_student = KMeans(n_clusters=student_n_cluster, random_state=0).fit(
        X)  # creates 5 clusters of the angles from the student video because theres 5 in student_n_cluster variable
    print(kmeans_student.labels_)
    print(kmeans_student.cluster_centers_)

    # n_cluster_coach = kmean_hyper_param_tuning(coach_frame)
    n_cluster_coach = 16  # Variable for Kmeans to see how many clusters to make
    X = np.array(coach_frames)
    kmeans_coach = KMeans(n_clusters=n_cluster_coach).fit(
        X)  # creates 11 clusters of the angles from the coach video because theres 5 in coach'_n_cluster variable
    print(n_cluster_coach)
    print(kmeans_coach.labels_)

    frame_range = [(0, 0)]
    for i in range(1, len(kmeans_coach.labels_)):
        if kmeans_coach.labels_[i] != kmeans_coach.labels_[i - 1]:
            if i - frame_range[-1][1] > 3:
                frame_range.append((frame_range[-1][1], i + 1))
            else:
                frame_range[-1] = (frame_range[-1][0], i + 1)

    frame_range = frame_range[1:]
    print(frame_range)


    # from IPython.display import Image, display
    #
    # for frame in frame_range:
    #     mid = (frame[0] + frame[1]) // 2
    #     display(Image(f'coach/coach{mid}.jpg')

    student_cluster = []
    start = 0
    for i in range(1, len(kmeans_student.labels_)):
        if kmeans_student.labels_[i] != kmeans_student.labels_[i - 1]:
            if i - 1 - start > 2:
                student_cluster.append({'label': kmeans_student.labels_[i - 1], 'start': start, 'end': i - 1})
            else:
                student_cluster[-1]['end'] = i - 1
            start = i
        else:
            student_cluster.append({'label': kmeans_student.labels_[i], 'start': start, 'end': i})

    print(student_cluster)
    print(len(student_cluster))
    Errors = []
    ErrorTotal = 0
    for label in student_cluster:
        index_student = (label['start'] + label['end']) // 2

        predict = kmeans_coach.predict([student_frames[index_student]])
        indexes_frame = np.where(kmeans_coach.labels_ == predict[0])

        nearest = get_nearest_neighbor(student_frames[index_student], indexes_frame[0], coach_frames)
        imageStudent = cv2.imread(f'student/student{index_student + 1}.jpg')
        imageCoach = cv2.imread(f'coach/coach{nearest}.jpg')
        print(index_student + 1, '******************', nearest)
        for i in range(1, len(student_frames[index_student]) - 1):
            ErrorMeasurement = student_frames[index_student][i] - coach_frames[nearest][i]
            Errors.append(ErrorMeasurement)
        for i in range(len(Errors)):
            if abs(Errors[i]) > 5:
                ErrorTotal += abs(Errors[i])
        ErrorTotal2 = ErrorTotal / len(Errors)
        Lettergrade = ""
        if ErrorTotal2 < 11:
            Lettergrade = "A"
        if ErrorTotal2 > 12 and ErrorTotal2 < 15:
            Lettergrade = "B"
        if ErrorTotal2 > 15 and ErrorTotal2 < 17:
            Lettergrade = "C"
        if ErrorTotal2 > 17:
            Lettergrade = "D or F"
        print(ErrorTotal2)
        #cv2.imshow('student', imageStudent)
        #cv2.imshow('coach', imageCoach)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        # for i in range(len(student_frames)):
        #    for i in range(len(student_frames[0])):
        return (Lettergrade)

'''
There is one small list for each frame containing all the angles for that frame, and there are multiple frames, 
meaning each frame has its own list in the megalist for the video. We will then put the data of the angles onto a cluster graph,
we will assign each data point to a centroid depending on the similarities. The relative location matters, but the location between the other clusters don't matter,
there will be a megalist of frames for the student and one for the coach. We will compare the difference of the angles between the student and the coach.
Depending on the difference of angles between the student and the coach, we will give them their score.
'''
'''
Scoring System:
300-100 score = Good
600-300 score = So-So
1000-600 score = Bad
'''
