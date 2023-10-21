import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose Detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
tpose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define the joint connections
joint_connections = [
    ('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'),
    ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'),
    ('LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE'),
    ('RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE'),
    ('LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'),
    ('RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'),
    ('LEFT_ELBOW','LEFT_SHOULDER','LEFT_HIP'),('RIGHT_ELBOW','RIGHT_SHOULDER','RIGHT_HIP')
]


# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    angle_rad = np.arccos(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)))
    angle_deg = np.degrees(angle_rad)

    return angle_deg


# Function to annotate angles on an image
def annotate_angles(frame, joint_connections, landmarks, mp_pose):
    i = 1
    angles = []
    for joint_connection in joint_connections:
        joint1, joint2, joint3 = joint_connection
        point1 = [landmarks[getattr(mp_pose.PoseLandmark, joint1).value].x,
                  landmarks[getattr(mp_pose.PoseLandmark, joint1).value].y]
        point2 = [landmarks[getattr(mp_pose.PoseLandmark, joint2).value].x,
                  landmarks[getattr(mp_pose.PoseLandmark, joint2).value].y]
        point3 = [landmarks[getattr(mp_pose.PoseLandmark, joint3).value].x,
                  landmarks[getattr(mp_pose.PoseLandmark, joint3).value].y]

        angle = calculate_angle(point1, point2, point3)
        angle_text = f"{joint1}-{joint2}-{joint3} Angle: {angle:.2f} degrees"
        angles.append(angle)
        cv2.putText(frame, angle_text, (10, 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        i += 1

    return angles


# Function to compare angles and return the median difference
def compare_angles(previous_angles, current_angles):
    differences = [abs(curr - prev) for curr, prev in zip(current_angles, previous_angles)]
    median_difference = np.median(differences)
    return median_difference


# Function to run pose similarity comparison
def run_pose_similarity(video_file, camera_index, similarity_score_threshold=0.5):
    # Video capture for the pre-recorded video
    cap_video = cv2.VideoCapture(video_file)

    # Video capture for the live camera feed
    cap_camera = cv2.VideoCapture(camera_index)
    cap_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Initialize variables for pausing and previous video frame
    is_paused = False
    oret_video, oframe_video = None, None
    fc = 0
    while True:
        # Read a frame from the video file if it's not paused
        if not is_paused:
            ret_video, vframe = cap_video.read()
            oret_video, oframe_video = ret_video, vframe
        else:
            ret_video, vframe = oret_video, oframe_video

        # Read a frame from the camera
        ret_camera, cframe = cap_camera.read()

        if not ret_video or not ret_camera:
            break

        crgb_frame = cv2.cvtColor(cframe, cv2.COLOR_BGR2RGB)
        cresults = pose.process(crgb_frame)

        rgb_vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB) if vframe is not None else None
        if rgb_vframe is not None:
            vresults = tpose.process(rgb_vframe)

        if cresults.pose_landmarks:
            clandmarks = cresults.pose_landmarks.landmark
            cangles = annotate_angles(cframe, joint_connections, clandmarks, mp_pose)
            mp_drawing.draw_landmarks(cframe, cresults.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if vresults.pose_landmarks:
            vlandmarks = vresults.pose_landmarks.landmark
            vangles = annotate_angles(vframe, joint_connections, vlandmarks, mp_pose)
            mp_drawing.draw_landmarks(vframe, vresults.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Compare the angles and calculate the similarity score
        similarity = compare_angles(cangles, vangles)
        print(similarity)
        if fc%30 == 0:
            if similarity < 5:
                is_paused = True  # Pause the video if the score goes below the threshold
            else:
                is_paused = False  # Unpause the video

        if not is_paused:
            # Show video frame only if it's not paused
            cv2.imshow('Video Playback', vframe)

        cv2.imshow('Camera Feed', cframe)
        fc+=1
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
            break

    # Release the video file and camera
    cap_video.release()
    cap_camera.release()
    cv2.destroyAllWindows()


# Example usage:
run_pose_similarity(r"C:\Users\sagar\OneDrive\Pictures\Camera Roll\WIN_20231021_14_55_21_Pro.mp4", 0)
