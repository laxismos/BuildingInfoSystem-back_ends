from typing import Tuple, List, Optional

import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression
OBJ_LIMIT = 120
ObjectMat = Tuple[float, float, float, float]

class ExtendedLinearRegression(LinearRegression):
    def __init__(self, max_angle_dev=10):
        super().__init__()
        self.max_angle_dev = max_angle_dev

def cross_rate(p1: ObjectMat, p2: ObjectMat):
    x1, y1, w1, h1 = p1
    x2, y2, w2, h2 = p2
    x_range = min(x1 + w1 / 2, x2 + w2 / 2) - max(x1 - w1 / 2, x2 - w2 / 2)
    y_range = min(y1 + h1 / 2, y2 + h2 / 2) - max(y1 - h1 / 2, y2 - h2 / 2)
    return x_range / w1 + x_range / w2, y_range / h1 + y_range / h2, x_range

def auto_residual_threshold(points: List[ObjectMat], alpha: Optional[float]=0.5, beta: Optional[float]=-1) -> float:
    ar_points = np.array(points)
    x_mean = np.mean(ar_points[:, 2])
    x_std = np.std(ar_points[:, 2])
    return max(abs(alpha * x_mean + beta * x_std), 10)

def find_point(points: List[ObjectMat], point: ObjectMat) -> None | ObjectMat:
    """
    :param points: The yolo object list with [x_center, y_center, width, height] shape.
    :param point: A single yolo object.
    :return: Remove and return the target yolo object from the list.If not found, return None.
    """
    best = None
    for index, point2 in enumerate(points):
        if point2[1] - point2[3]/2 < point[1] - point[3]/2:
            continue
        p, q, r = cross_rate(point, point2)
        if p > 0.55 and q <= p and q < 1.2:
            if best is None:
                best = [point2, p, q, r]
            elif best[3] < r:
                p1, q1, r1 = cross_rate(best[0], point2)
                if q1 > 0.25:
                    best = [point2, p, q, r]

    if best is None:
        return best
    else:
        points.remove(best[0])
        return best[0]

def detect_columns(obj_list: List[ObjectMat], residual_threshold: Optional[float]=None, number_threshold: Optional[str]=None) -> List[List[ObjectMat]]:
    columns = []
    points = obj_list.copy()
    if residual_threshold is None:
        residual_threshold = auto_residual_threshold(points)
    if number_threshold is None:
        number_threshold = OBJ_LIMIT
    if len(points) > number_threshold:
        ransac = RANSACRegressor(
            estimator=ExtendedLinearRegression(15),
            residual_threshold=residual_threshold,
            max_trials=100
        )
        while len(points) > 3:
            copy_points = points[:]
            ar_points = np.array(points)
            X = ar_points[:, 1].reshape(-1, 1)
            y = ar_points[:, 0].reshape(-1, 1)
            predictor = ransac.fit(X, y)
            predictions = predictor.predict(X)
            column = []
            for i, prediction in enumerate(predictions):
                if abs(prediction - y[i]) < residual_threshold:
                    column.append(copy_points[i])
                    points.remove(copy_points[i])
            column.sort(key=lambda z: z[1], reverse=True)
            columns.append(column)
    else:
        points.sort(key=lambda z: z[1] - z[3]/2)
        while len(points) > 0:
            column = [points.pop(0)]
            # copy_points = points.copy()
            last_point = find_point(points, column[0])
            while last_point is not None:
                column.append(last_point)
                last_point = find_point(points, last_point)

            column.sort(key=lambda z: z[1] + z[3]/2, reverse=True)

            columns.append(column)

    return columns

# class Visualize:
#     def __init__(self, base_path: None | Path = None):
#         self.image_index = 0
#         self.dir_already_created = False
#         if base_path is None:
#             self.save_path = Path('./check/error/pred_result')
#         else:
#             self.save_path = base_path
#         self.id = 1
#
#     def create_dir(self):
#         if self.dir_already_created:
#             raise Exception('Directory already created.')
#         max_id = 0
#         for name in self.save_path.iterdir():
#             if name.is_dir():
#                 loc = name.stem.find('pred')
#                 if loc == 0:
#                     max_id = max(int(name.stem[4:]), max_id)
#         self.id = max_id + 1
#         new_dir = self.save_path / f'pred{self.id}'
#         new_dir.mkdir(parents=True, exist_ok=True)
#         self.save_path = new_dir
#         self.dir_already_created = True
#         correct_dir = self.save_path / 'correct'
#         error_dir = self.save_path / 'error'
#         correct_dir.mkdir(parents=True, exist_ok=True)
#         error_dir.mkdir(parents=True, exist_ok=True)
#
#     def write_image(self, image, cols: List[List[ObjectMat]], building=None, img_label:bool=None, extra_info:str=""):
#         color = [
#             (0, 0, 255), (0, 255, 0), (255, 0, 0),
#             (255, 255, 0), (255, 0, 255), (0, 255, 255),
#             (100, 0, 255), (255, 0, 100), (100, 255, 0),
#             (255, 100, 0), (0, 100, 255), (0, 255, 100),
#             (100, 100, 150), (50, 150, 100), (150, 50, 50)
#         ]
#         c_index = 0
#         for col in cols:
#             c_index = (c_index + 1) % len(color)
#             for p in col:
#                 x, y, w, h = p
#                 cv2.rectangle(image, [int(x - w / 2), int(y - h / 2)], [int(x + w / 2), int(y + h / 2)], color[c_index], 2)
#         if building is not None:
#             x, y, w, h = building
#             cv2.rectangle(image, [int(x - w / 2), int(y - h / 2)], [int(x + w / 2), int(y + h / 2)], (15, 15, 220), 2)
#         img_name = str(self.image_index) + extra_info +  '.jpg'
#         if img_label is not None:
#             if img_label:
#                 t = "correct"
#             else:
#                 t = "error"
#             t_path = self.save_path / t / img_name
#             cv2.imwrite(str(t_path), image)
#         else:
#             cv2.imwrite(str(self.save_path / img_name), image)
#         self.image_index += 1

def get_main_building(buildings) -> ObjectMat:
    central_building = None
    org_y, org_x = buildings.orig_shape
    total_area = org_x * org_y
    for building in buildings.boxes.xywh:
        x, y, w, h = building.cpu().numpy()
        area = w * h
        if area > total_area * 0.12:
            if central_building is None:
                central_building = (x, y, w, h)
            else:
                x1, y1, w1, h1 = central_building
                if np.sqrt((x1 - org_x / 2) ** 2 + (y1 - org_y / 2) ** 2) > np.sqrt(
                        (x - org_x / 2) ** 2 + (y - org_y / 2) ** 2):
                    central_building = (x, y, w, h)
    return central_building

def exclude_points(building, result):
    points = []
    for i, box in enumerate(result.boxes.xywh):
        x, y, w, h = box.cpu().numpy()
        if building is not None:
            wid = min(x + w / 2, building[0] + building[2] / 2) - max(x - w / 2, building[0] -
                                                                                      building[2] / 2)
            hei = min(y + h / 2, building[1] + building[3] / 2) - max(y - h / 2, building[1] -
                                                                                      building[3] / 2)
            if wid > 0 and hei > 0 and wid * hei > w * h * 0.9:
                points.append((x, y, w, h))
        else:
            points.append((x, y, w, h))
    return points



