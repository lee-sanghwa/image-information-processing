import cv2
import numpy as np
import dlib


# 이미지의 크기를 width 와 height를 동일한 비율로 조절하는 함수
def resize_image_with_percent(image, percent):
    width = int(image.shape[1] * percent / 100)
    height = int(image.shape[0] * percent / 100)
    temp_image = cv2.resize(image, dsize=(width, height))

    return temp_image


# landmarks들로 부터 x, y 좌표를 얻어내는 함수
def get_points_from_landmarks(landmarks):
    list_of_landmarks_points = []

    for i in range(0, 68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y

        list_of_landmarks_points.append((x, y))

    return list_of_landmarks_points


# 얼굴 테두리안에서 landmark를 통해 들로네 삼각분할 진행하는 함수
def delaunay_triangulation_with_face_hull_and_landmarks(face_hull, numpy_of_landmarks_points):
    # 들로네 삼각분할을 하기 위해서는 특정 공간 안에서만 삼각형이 만들어져야하므로, 얼굴 테두리를 포함하는 사각형을 그 특정공간으로 쓰이도록 함.
    face_rectangular = cv2.boundingRect(face_hull)
    face_subdiv = cv2.Subdiv2D(face_rectangular)
    tuple_of_landmarks_points = map(tuple, numpy_of_landmarks_points)

    for tuple_of_landmarks_point in tuple_of_landmarks_points:
        face_subdiv.insert(tuple_of_landmarks_point)

    triangles = face_subdiv.getTriangleList()
    triangles = np.array(triangles, np.int32)

    return triangles


# 삼각형을 이루는 좌표들을 landmark의 index로 구해주는 함수
def get_triangles_landmark_indexes(triangles, np_of_landmarks_points):
    triangles_landmark_indexes = []

    # triangle : [165 165 207 174 194 187] : [x1 y1 x2 y2 x3 y3]
    for i, triangle in enumerate(triangles):
        angle1_point = (triangle[0], triangle[1])
        angle2_point = (triangle[2], triangle[3])
        angle3_point = (triangle[4], triangle[5])

        # angle1_landmark_index => (array([17]),) -> array([17]) -> 17
        angle1_landmark_index = np.where((np_of_landmarks_points == angle1_point).all(axis=1))[0][0]
        angle2_landmark_index = np.where((np_of_landmarks_points == angle2_point).all(axis=1))[0][0]
        angle3_landmark_index = np.where((np_of_landmarks_points == angle3_point).all(axis=1))[0][0]

        triangles_landmark_indexes.append((angle1_landmark_index, angle2_landmark_index, angle3_landmark_index))

    return triangles_landmark_indexes


# 얼굴을 바꾸기 위해 필요한 정보들을 return 하는 함수
def get_information_for_change_face(ori_image, gray_image, is_monkey, triangles_landmark_indexes=None):
    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

    # 얼굴만을 얻을 때 사용하기 위한 mask
    mask_for_getting_face = np.zeros_like(gray_image)
    # 얼굴만을 없앨 때 사용하기 위한 mask
    mask_for_delete_face = np.ones_like(gray_image)
    # 원점을 기준으로했을 때의 삼각형들의 좌표를 저장할 공간
    triangle_points_with_zero_origin_value_list = []
    # 삼각형의 내부의 이미지만을 갖고 있는 사각형에서의 이미지 픽셀값들 (삼각형 외부는 검정색)
    rectangulars_include_only_inside_triangle = []
    # 삼각형에 외접하는 사각형의 맨 왼쪽 위의 좌표와 너비, 높이에 대한 정보를 저장할 공간
    rectangulars_for_triangle = []

    # gray image에서 얼굴들을 추출
    faces = face_detector(gray_image)
    # 현재는 처음으로 추출된 얼굴만을 다룰 예정
    face = faces[0]
    # gray image에서 발견된 얼굴에 대해 이미 학습된 데이터를 통해 68개의 특징 점을 추출
    landmarks = face_predictor(gray_image, face)
    # 각 landmark에 대한 x, y 좌표를 추출
    list_of_landmarks_points = get_points_from_landmarks(landmarks)

    # list를 numpy array로 변환하여 연산속도를 빠르게 할 수 있도록 함
    np_of_landmarks_points = np.array(list_of_landmarks_points, np.int32)

    # 랜드마크들 중에서 얼굴의 테두리를 나타내는 점들을 찾아내기 위해 convexHull() 이용
    face_hull = cv2.convexHull(np_of_landmarks_points)

    # 얼굴만을 얻기위해 얼굴의 테두리 안쪽만 255로 채움
    cv2.fillConvexPoly(mask_for_getting_face, face_hull, 255)
    face_image = cv2.bitwise_and(ori_image, ori_image, mask=mask_for_getting_face)

    # 얼굴을 제외한 나머지 부분을 얻기위해 얼굴의 테두리 안쪽만 0으로 채움
    cv2.fillConvexPoly(mask_for_delete_face, face_hull, 0)
    without_face_image = cv2.bitwise_and(ori_image, ori_image, mask=mask_for_delete_face)

    # monkey image의 경우
    if is_monkey:
        # 들로네 삼각분할을 통해 landmarks에서의 삼각형들 추출
        triangles = delaunay_triangulation_with_face_hull_and_landmarks(face_hull, np_of_landmarks_points)

        # 각 삼각형이 어떤 landmark 들을 통해 이루어졌는가를 추출.
        triangles_landmark_indexes = get_triangles_landmark_indexes(triangles, np_of_landmarks_points)
    # else -> target image의 경우를 의미 -> triangles_landmark_indexes를 입력받음. 따라서 들로네 삼각분할을 통해 구할 필요 없음.

    for i, triangle_landmark_indexes in enumerate(triangles_landmark_indexes):
        angle1_point = list_of_landmarks_points[triangle_landmark_indexes[0]]
        angle2_point = list_of_landmarks_points[triangle_landmark_indexes[1]]
        angle3_point = list_of_landmarks_points[triangle_landmark_indexes[2]]

        triangle_points = np.array([angle1_point, angle2_point, angle3_point])

        # 삼각형을 포함하는 직사각형의 왼쪽 맨 위의 x, y 좌표와, 너비, 높이를 추출.
        (x, y, w, h) = cv2.boundingRect(triangle_points)
        rectangulars_for_triangle.append((x, y, w, h))

        """ 삼각형 내부의 컬러 이미지만을 얻고, 외부는 검정색으로 하기위한 과정 """
        rectangular_for_triangle = ori_image[y: y + h, x: x + w]
        mask_of_rectangular_for_triangle = np.zeros((h, w), np.uint8)

        triangle_points_with_zero_origin_value = np.array([
            [angle1_point[0] - x, angle1_point[1] - y],
            [angle2_point[0] - x, angle2_point[1] - y],
            [angle3_point[0] - x, angle3_point[1] - y]
        ], np.int32)

        triangle_points_with_zero_origin_value_list.append(triangle_points_with_zero_origin_value)

        cv2.fillConvexPoly(mask_of_rectangular_for_triangle, triangle_points_with_zero_origin_value, 255)

        cropped_triangle = cv2.bitwise_and(rectangular_for_triangle, rectangular_for_triangle,
                                           mask=mask_of_rectangular_for_triangle)
        """ 삼각형 내부의 컬러 이미지만을 얻고, 외부는 검정색으로 하기위한 과정 끝 """
        rectangulars_include_only_inside_triangle.append(cropped_triangle)

    return face_image, triangle_points_with_zero_origin_value_list, rectangulars_include_only_inside_triangle, \
        rectangulars_for_triangle, triangles_landmark_indexes, without_face_image


if __name__ == '__main__':
    monkey_image = cv2.imread("./images/monkey.jpeg")
    # monkey 이미지 크기가 커서 90%로 크기 조정
    monkey_image = resize_image_with_percent(monkey_image, 90)
    # 영상처리를 위해 색깔 성분 제거한 monkey 이미지
    gray_monkey_image = cv2.cvtColor(monkey_image, cv2.COLOR_BGR2GRAY)

    target_image = cv2.imread("./images/person.png")
    # target 이미지 크기가 커서 크기 조정
    target_image = resize_image_with_percent(target_image, 50)

    # target_image = cv2.imread("./images/person1.jpeg")
    # # target 이미지 크기가 커서 크기 조정
    # target_image = resize_image_with_percent(target_image, 150)

    # 나중에 합성을 위해 복사
    copied_target_image = np.zeros_like(target_image)
    # 영상처리를 위해 색깔 성분 제거한 monkey 이미지
    gray_target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    (monkey_face_image,
     monkey_triangle_points_with_zero_origin_value_list,
     monkey_cropped_triangle_list,
     monkey_rectangular_for_triangle_list,
     monkey_triangle_indexes,
     monkey_without_face_image) = get_information_for_change_face(monkey_image, gray_monkey_image, True)

    (target_face_image,
     target_triangle_points_with_zero_origin_value_list,
     target_cropped_triangle_list,
     target_rectangular_for_triangle_list,
     target_triangle_indexes,
     target_without_face_image) = get_information_for_change_face(target_image, gray_target_image, False,
                                                                  monkey_triangle_indexes)

    for index, monkey_triangle_points in enumerate(monkey_triangle_points_with_zero_origin_value_list):
        monkey_triangle_points = np.float32(monkey_triangle_points)
        target_triangle_points = np.float32(target_triangle_points_with_zero_origin_value_list[index])
        target_x = target_rectangular_for_triangle_list[index][0]
        target_y = target_rectangular_for_triangle_list[index][1]
        target_w = target_rectangular_for_triangle_list[index][2]
        target_h = target_rectangular_for_triangle_list[index][3]

        # 원숭이 사진의 삼각형을 target의 삼각형에 맞게 아핀변환
        M = cv2.getAffineTransform(monkey_triangle_points, target_triangle_points)
        warped_triangle = cv2.warpAffine(monkey_cropped_triangle_list[index], M, (target_w, target_h))

        # target 이미지를 복사한 곳에 직접 warped_triangle 를 넣게되면 몇몇 삼각형 조각들이 제대로 들어가지지않아 cv2.add()를 이용하여 넣음
        triangle_area = copied_target_image[target_y: target_y + target_h, target_x: target_x + target_w]
        triangle_area = cv2.add(triangle_area, warped_triangle)

        copied_target_image[target_y: target_y + target_h, target_x: target_x + target_w] = triangle_area

    # target 이미지에서의 얼굴이 검정색으로 된 이미지와 target 이미지에서의 얼굴과 유사하게 볌형한 원숭이 얼굴을 합침.
    target_image_with_monkey_face = cv2.add(target_without_face_image, copied_target_image)

    cv2.imshow("Monkey Image", monkey_image)
    cv2.imshow("Target Image", target_image)
    cv2.imshow("Target Image With Monkey Face", target_image_with_monkey_face)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
