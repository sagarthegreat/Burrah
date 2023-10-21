import cv2
import mediapipe as mp
import numpy as np


def pose_similarity(detected_pose, reference_pose):
    distances = np.linalg.norm(detected_pose - reference_pose, axis=1)
    return np.mean(distances)


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
# Open the video file for playback
video_file = r"C:\Users\sagar\iCloudPhotos\Photos\RPReplay_Final1695854330.mp4"
cap_video = cv2.VideoCapture(video_file)

# Open the webcam for live camera feed
cap_camera = cv2.VideoCapture(0)  # 0 represents the default camera

while True:
    # Read a frame from the video file
    ret_video, frame_video = cap_video.read()

    # Read a frame from the camera
    ret_camera, frame_camera = cap_camera.read()

    if not ret_video or not ret_camera:
        break

    rgb_cframe = cv2.cvtColor(frame_camera, cv2.COLOR_BGR2RGB)
    cresults = pose.process(rgb_cframe)

    rgb_vframe = cv2.cvtColor(frame_video, cv2.COLOR_BGR2RGB)
    vresults = pose.process(rgb_vframe)

    if cresults.pose_landmarks is not None and vresults.pose_landmarks is not None:
        cpose = np.array([[lm.x, lm.y] for lm in cresults.pose_landmarks.landmark])
        vpose = np.array([[lm.x, lm.y] for lm in vresults.pose_landmarks.landmark])
        similarity_score = pose_similarity(cpose, vpose)
        print(f'Similarity Score: {similarity_score}')


    cv2.imshow('Camera Feed', frame_camera)
    # Display the video frame in a window
    cv2.imshow('Video Playback', frame_video)

    # Display the camera frame in a separate window


    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
        break

# Release the video file and camera
cap_video.release()
cap_camera.release()
cv2.destroyAllWindows()
