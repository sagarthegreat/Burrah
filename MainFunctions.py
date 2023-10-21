import cv2
import mediapipe as mp
import numpy as np

def pose_similarity(detected_pose, reference_pose):
    distances = np.linalg.norm(detected_pose - reference_pose, axis=1)
    return np.mean(distances)

def run_pose_similarity(video_file, camera_index, similarity_score_threshold=0.5):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Open the video file for playback
    cap_video = cv2.VideoCapture(video_file)

    # Open the webcam for live camera feed
    cap_camera = cv2.VideoCapture(camera_index)  # 0 represents the default camera

    # Track the pause state and previous video frame
    is_paused = False
    oret_video, oframe_video = None, None

    while True:
        # Read a frame from the video file if it's not paused
        if not is_paused:
            ret_video, frame_video = cap_video.read()
            oret_video, oframe_video = ret_video, frame_video
        else:
            ret_video, frame_video = oret_video, oframe_video

        # Read a frame from the camera
        ret_camera, frame_camera = cap_camera.read()

        if not ret_video or not ret_camera:
            break

        rgb_cframe = cv2.cvtColor(frame_camera, cv2.COLOR_BGR2RGB)
        cresults = pose.process(rgb_cframe)

        rgb_vframe = cv2.cvtColor(frame_video, cv2.COLOR_BGR2RGB) if frame_video is not None else None

        if rgb_vframe is not None:
            vresults = pose.process(rgb_vframe)

        if cresults.pose_landmarks is not None and vresults.pose_landmarks is not None:
            cpose = np.array([[lm.x, lm.y] for lm in cresults.pose_landmarks.landmark])
            vpose = np.array([[lm.x, lm.y] for lm in vresults.pose_landmarks.landmark])
            similarity_score = pose_similarity(cpose, vpose)
            print(f'Similarity Score: {similarity_score}')

            # Draw landmarks and lines on the camera frame
            try:
                for lm in cresults.pose_landmarks.landmark:
                    h, w, _ = frame_camera.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame_camera, (cx, cy), 5, (0, 255, 0), -1)

                # Draw landmarks and lines on the video frame
                for lm in vresults.pose_landmarks.landmark:
                    h, w, _ = frame_video.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame_video, (cx, cy), 5, (0, 255, 0), -1)

                # Connect landmarks with lines
                for connection in mp_pose.POSE_CONNECTIONS:
                    first_landmark = cresults.pose_landmarks.landmark[connection[0]]
                    second_landmark = cresults.pose_landmarks.landmark[connection[1]]
                    x1, y1 = int(first_landmark.x * w), int(first_landmark.y * h)
                    x2, y2 = int(second_landmark.x * w), int(second_landmark.y * h)
                    cv2.line(frame_camera, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    first_landmark = vresults.pose_landmarks.landmark[connection[0]]
                    second_landmark = vresults.pose_landmarks.landmark[connection[1]]
                    x1, y1 = int(first_landmark.x * w), int(first_landmark.y * h)
                    x2, y2 = int(second_landmark.x * w), int(second_landmark.y * h)
                    cv2.line(frame_video, (x1, y1), (x2, y2), (0, 255, 0), 2)
            except:
                pass

            if similarity_score < similarity_score_threshold:
                is_paused = True  # Pause the video if the score goes below the threshold
            else:
                is_paused = False  # Unpause the video

        if not is_paused:
            # Show video frame only if it's not paused
            cv2.imshow('Video Playback', frame_video)

        cv2.imshow('Camera Feed', frame_camera)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
            break

    # Release the video file and camera
    cap_video.release()
    cap_camera.release()
    cv2.destroyAllWindows()

# Example usage:
run_pose_similarity(r"C:\Users\sagar\iCloudPhotos\Photos\RPReplay_Final1695854330.mp4", 0)
