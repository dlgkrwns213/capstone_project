import mediapipe as mp
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from scipy.spatial import ConvexHull

class MultiModelDetector:
    def __init__(self, behavior='all'):
        bi = Behavior_Information(behavior)
        hand_detection, hand_tracking, face_detection, face_tracking, pose_detection, pose_tracking = bi.prob

        # hand
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2 if hand_detection <= 1 else 0,
            min_detection_confidence=hand_detection,
            min_tracking_confidence=hand_tracking
        )

        # face
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1 if face_detection <= 1 else 0,
            min_detection_confidence=face_detection,
            min_tracking_confidence=face_tracking
        )
        self.cheeks_idx = {147, 177, 187, 192, 207, 213, 214, 215, 376, 401, 411, 416, 427, 433, 434, 435}

        # pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=pose_detection,
            min_tracking_confidence=pose_tracking
        )

        self.idx1, self.idx2 = self.get_idx()
        self.vector_idx1, self.vector_idx2 = self.get_vector_idx()
        self.behaviors = bi.behaviors
        self.num_classes = len(self.behaviors)

    
    def find_body(self, cv2_image):
        '''
        When an image is inputted using a mediapipe, the function finds two hands, face, and pose and adds them to the image

        >>> MultiModelDetector.find_body(cv2_image)
        mediapiped_cv2_image
        '''
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        img = cv2_image.copy()
        imgRGB = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

        # find hand
        hands_results = self.hands.process(imgRGB)
        if hands_results.multi_hand_landmarks is not None:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img,
                          hand_landmarks,
                          self.mp_hands.HAND_CONNECTIONS,
                          mp_drawing_styles.get_default_hand_landmarks_style(),
                          mp_drawing_styles.get_default_hand_connections_style())

        # find face
        face_mesh_results = self.face_mesh.process(imgRGB)
        if face_mesh_results.multi_face_landmarks is not None:
            for face_mesh_landmarks in face_mesh_results.multi_face_landmarks:
                cheeks = [v for i, v in enumerate(face_mesh_landmarks.landmark) if i in self.cheeks_idx]
                ci = landmark_pb2.NormalizedLandmarkList()
                ci.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in cheeks])
                mp_drawing.draw_landmarks(img, ci) 

        # find pose
        pose_results = self.pose.process(imgRGB)
        if pose_results.pose_landmarks is not None:
            mp_drawing.draw_landmarks(img,
                                      pose_results.pose_landmarks,
                                      self.mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        return img

    def image_get_location(self, image):
        '''
        Returns each position value of two hands, face, and pose present in the image when an image is inputted using a media pipe

        >>> locations: nd.ndarray = MultiModelDetector.image_get_location(cv2_image)
        >>> locations.shape
        (91, 3)
        '''
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        hands_results = self.hands.process(imgRGB)
        face_mesh_results = self.face_mesh.process(imgRGB)
        pose_results = self.pose.process(imgRGB)

        locations = np.zeros((91, 3))

        # no pose -> error
        if pose_results.pose_landmarks is None:
            return np.zeros((91, 3))

        # find pose -> 33
        for i, lm in enumerate(pose_results.pose_landmarks.landmark):
            locations[58+i] = [lm.x, lm.y, lm.z]
            
        # find hand -> 21 * 2
        if hands_results.multi_hand_landmarks is not None:
            if pose_results.pose_landmarks is not None:
                pose_landmarks = pose_results.pose_landmarks.landmark
                left_wrist, right_wrist = pose_landmarks[15], pose_landmarks[16]
                
                left_wrist_array = np.array([left_wrist.x, left_wrist.y])
                right_wrist_array = np.array([right_wrist.x, right_wrist.y])

                for hand_landmarks in hands_results.multi_hand_landmarks:
                    wrist = hand_landmarks.landmark[0]
                    wrist_array = np.array([wrist.x, wrist.y])

                    # print(left_wrist_array, right_wrist_array)
                    # print(wrist_array)
                    
                    distances_left_wrist = np.linalg.norm(left_wrist_array - wrist_array)
                    distances_right_wrist = np.linalg.norm(right_wrist_array - wrist_array)

                    # print(distances_left_wrist, distances_right_wrist)
                    if distances_left_wrist < distances_right_wrist:
                        x = 21
                        # print('is_left')
                    else:
                        x = 0
                        # print('is_right')

                    for i, lm in enumerate(hand_landmarks.landmark):
                        locations[x+i] = [lm.x, lm.y, lm.z]

        # find face -> 16
        if face_mesh_results.multi_face_landmarks is not None:
            for face_mesh_landmarks in face_mesh_results.multi_face_landmarks:
                cheeks = [v for i, v in enumerate(face_mesh_landmarks.landmark) if i in self.cheeks_idx]
                for i, lm in enumerate(cheeks):
                    locations[42+i] = [lm.x, lm.y, lm.z]            

        return locations

    def get_idx(self):
        right_hand1 = [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
        right_hand2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11, 12, 13,14, 15, 16, 17,18, 19, 20]
        
        left_hand1 = [21, 22, 23, 24, 21, 26, 27, 28, 21, 30, 31, 32, 21, 34, 35, 36, 21, 38, 39, 40]
        left_hand2 = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
        
        pose1 = [69, 69, 71, 73, 73, 73, 70, 72, 74, 74, 74, 69, 70, 81, 81, 83, 82, 84]
        pose2 = [70, 71, 73, 75, 77, 79, 72, 74, 76, 78, 80, 81, 82, 82, 83, 85, 84, 86]
        
        finger = [8, 8, 8, 8, 8, 8, 8, 8, 29, 29, 29, 29, 29, 29, 29, 29]
        cheek =  [42,43,44,45,46,47,48,49,50, 51, 52, 53, 54, 55, 56, 57]
        
        return right_hand1 + left_hand1 + pose1 + finger, right_hand2 + left_hand2 + pose2 + cheek

    def get_vector_idx(self):
        right_hand_vector1 = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
        right_hand_vector2 = [1, 2, 3, 5, 6, 7, 9, 10,11, 13, 14, 15, 17, 18, 19]
        
        left_hand_vector1 = [20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38]
        left_hand_vector2 = [21, 22, 23, 25, 26, 27, 29, 30, 31, 33, 34, 35, 37, 38, 39]
        
        pose_vector1 = [40, 41, 42, 42, 42, 40, 46, 47, 47, 47, 40, 40, 52, 53, 53, 56]
        pose_vector2 = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 57]
        
        finger_vector = [7, 7, 7, 7, 7, 7, 7, 7, 27, 27, 27, 27, 27, 27, 27, 27]
        cheek_vector =  [58,59,60,61,62,63,64,65,66, 67, 68, 69, 70, 71, 72, 73]
        
        return right_hand_vector1 + left_hand_vector1 + pose_vector1 + finger_vector, right_hand_vector2 + left_hand_vector2 + pose_vector2 + cheek_vector

    def get_angle_numpy(self, locations):
        '''
        Return the calculated value of the angle of each joint when the body position is input

        >>> locations: nd.ndarray = MultiModelDetector.image_get_location(cv2_image)
        >>> angles: nd.ndarray = MultiModelDetector.get_angle_numpy(cv2_image)
        >>> angles.shape
        (62,)
        '''
        v = locations[self.idx1, :] - locations[self.idx2, :]
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]  # Normailize v (v의 길이를 1로 만듦)

        angle = np.arccos(np.einsum('nt,nt->n',
                                    v[self.vector_idx1, :],
                                    v[self.vector_idx2, :]))
        angle = np.degrees(angle)
        angle[np.isnan(angle)] = 0
        return angle

    def get_distances_numpy(self, locations):
        '''
        Return the proportion of the body using the position of the pose
        
        >>> locations: nd.ndarray = MultiModelDetector.image_get_location(cv2_image)
        >>> distances: nd.ndarray = MultiModelDetector.get_distances_numpy(locations)
        >>> distances.shape
        (12,)
        '''
        distances_rate = np.zeros(12)
        standard_distance = np.linalg.norm(locations[58] - locations[86])
        if standard_distance == 0:
            return distances_rate
        idx_to_check = [83, 81, 75, 73, 71, 69, 84, 82, 76, 74, 72, 70]
        distances_rate[:6] = np.linalg.norm(locations[idx_to_check[:6]] - locations[85], axis=1)
        distances_rate[6:] = np.linalg.norm(locations[idx_to_check[6:]] - locations[86], axis=1)
        return distances_rate / standard_distance

    def get_meet_finger_cheek(self, locations):
        '''
        Use the coordinates of the hand and face to return whether the coordinates of the index finger exist within the coordinates of the face

        >>> locations: nd.ndarray = MultiModelDetector.image_get_location(cv2_image)
        >>> is_meets: nd.ndarray = MultiModelDetector.get_meet_finget_cheek(locations)
        >>> is_meets.shape
        (2,)
        '''
        
        def point_in_polygon(x, y, polygon):
            if any(np.array_equal(polygon[i], np.array([0.0, 0.0])) for i in range(7)):
                return False

            hull = ConvexHull(polygon)
            return all(np.dot(hull.equations[:, :-1], [x, y]) + hull.equations[:, -1] <= 0)

        right_finger, left_finger = locations[8], locations[29]
        right_cheeks_xy = np.array([locations[42], locations[43], locations[49], locations[45], locations[48], locations[46], locations[44]])[:, :2]
        left_cheeks_xy = np.array([locations[50], locations[51], locations[57], locations[53], locations[56], locations[54], locations[52]])[:, :2]

        is_meet_right = int(point_in_polygon(*right_finger[:2], right_cheeks_xy))
        is_meet_left = int(point_in_polygon(*left_finger[:2], left_cheeks_xy))
        return np.array([is_meet_right, is_meet_left])
    
    def image_to_numpy(self, image):
        '''
        Returns the value of the numpy array to be learned when the image is entered

        >>> datas: nd.ndarray = MultiModelDetector.image_to_numpy(cv2_image)
        >>> datas.shape
        (76,)
        '''
        locations = self.image_get_location(image)
        angles = self.get_angle_numpy(locations)
        distances = self.get_distances_numpy(locations)
        cheek_meet = self.get_meet_finger_cheek(locations)
        return np.concatenate((angles, distances, cheek_meet))


class Behavior_Information:
    def __init__(self, behavior=''):
        version = str2int[behavior]
        if version == 0:
            self.prob = 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
        elif version == 1:  # rollhand, rollfeet, hooray, jumphooray -> pose
            self.prob = 1.1, 0, 1.1, 0, 0.7, 0.7
        elif version == 2: # clap -> hand & pose
            self.prob = 0.5, 0.5, 1.1, 1.1, 0.7, 0.7
        elif version == 3: # cheek -> hand & face
            self.prob = 0.5, 0.5, 0.5, 0.5, 0.8, 0.8
        elif version == 4:  # RPS, thumbs_up -> hand, pose
            self.prob = 0.5, 0.5, 1.1, 1.1, 0.8, 0.8
        elif version == 5:
            self.prob = 0.5, 0.01, 0.5, 0.01, 0.7, 0.01

        self.behaviors = list(get_label[behavior_idx[behavior]].keys())


behaviors = ['rollhand', 'rollfeet', 'hooray', 'jumphooray',
             'clap', 'cheek', 'RPS', 'thumbsup', 'all', 'find_image']
str2int = {'rollhand': 1, 'rollfeet': 1, 'hooray': 1, 'jumphooray': 1,
           'clap': 2, 'cheek': 3, 'RPS': 4, 'thumbsup': 4, 'all': 0, 'find_image': 5}
behavior_idx = {'rollhand': 0, 'rollfeet': 1, 'hooray': 2, 'jumphooray': 3,
                'clap': 4, 'cheek': 5, 'RPS': 6, 'thumbsup': 7, 'all': 8, 'find_image': 9}
get_label = [{'None': 0, 'rollhand1': 1, 'rollhand2': 2},
             {'None': 0, 'rollfeet1': 1, 'rollfeet2': 2},
             {'None': 0, 'hooray': 1}, 
             {'None': 0, 'jumphooray': 1, 'hooray': 2},
             {'None': 0, 'clap1': 1, 'clap2': 2},
             {'None': 0, 'cheek_left': 1, 'cheek_right': 2},
             {'Hand_None': 0, 'rock_left': 1, 'rock_right': 2, 'paper_left': 3, 'paper_right': 4, 
              'scissors_left': 5, 'scissors_right': 6, 'scissors23_left': 7, 'scissors23_right': 8},
             {'Hand_None': 0, 'thumbsup_both': 1, 'thumbsup_left': 2, 'thumbsup_right': 3},
             {'None': 0, 'cheek_left': 1, 'cheek_right': 2, 'clap1': 3, 'clap2': 4, 
              'hooray': 5, 'jumphooray': 6, 'paper_left': 7, 'paper_right': 8, 'rock_left': 9,
              'rock_right': 10, 'rollfeet1': 11, 'rollfeet2': 12, 'rollhand1': 13, 'rollhand2': 14,
              'scissors_left': 15, 'scissors_right': 16, 'thumbsup_both': 17, 'thumbsup_left': 18, 'thumbsup_right': 19,
              'scissors23_left': 20, 'scissors23_right': 21},
             {'None': 0, 'cheek_left': 1, 'cheek_right': 2, 'clap1': 3, 'clap2': 4, 
              'hooray': 5, 'jumphooray': 6, 'paper_left': 7, 'paper_right': 8, 'rock_left': 9,
              'rock_right': 10, 'rollfeet1': 11, 'rollfeet2': 12, 'rollhand1': 13, 'rollhand2': 14,
              'scissors_left': 15, 'scissors_right': 16, 'thumbsup_both': 17, 'thumbsup_left': 18, 'thumbsup_right': 19,
              'scissors23_left': 20, 'scissors23_right': 21}]