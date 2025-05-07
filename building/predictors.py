import json
from os import PathLike

import torch
from torch import nn
from django.http import JsonResponse, HttpRequest
from ultralytics import YOLO
from .floor_recognition import *
from torchvision import models, transforms
import cv2

BATCH_SIZE = 16

class BasePredictor(object):
    def __init__(self):
        self.last_image = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = BATCH_SIZE
        pass

    def predict(self, img=None)->List[str]:
        raise NotImplementedError

    @classmethod
    def get_image_from_request(cls, request: HttpRequest):
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image file provided'}, status=400), None

        image_files = request.FILES.getlist('image')
        file_names = []
        images = []
        for image_file in image_files:
            if image_file.content_type != 'image/jpeg':
                continue
            file_names.append(image_file.name)
            byte_image = image_file.read()
            if byte_image is None:
                raise Exception('Image has been loaded!')
            image = cv2.imdecode(np.asarray(bytearray(byte_image), dtype=np.uint8), cv2.IMREAD_COLOR)
            images.append(image)

        return images, file_names

    def transform_image(self, image_list, trans):
        if not isinstance(trans, transforms.Compose):
            raise TypeError('trans should be a Compose object')
        if not isinstance(image_list, list):
            raise TypeError('image_list should be a list')
        image_tensor_list = []
        for image in image_list:
            image_tensor_list.append(trans(image))
        return torch.stack(image_tensor_list).to(self.device)

class FloorPredictor(BasePredictor):
    def __init__(self, building_model_path, outer_obj_model_path):
        super().__init__()
        self.__building_model = YOLO(building_model_path).eval()
        self.__outer_obj_model = YOLO(outer_obj_model_path).eval()

    def predict(self, images=None):
        if images is None:
            images = self.last_image
        floors_list = []
        for img in images:
            buildings = self.__building_model.predict(
                source=img,
                imgsz=640,
                save=False,
                conf=0.5,
                iou=0.55,
                show=False,
                verbose=False
            )
            outer_objects = self.__outer_obj_model.predict(
                source=img,
                imgsz=960,
                save=False,
                conf=0.5,
                iou=0.6,
                show=False,
                verbose=False
            )
            main_building = get_main_building(buildings[0])
            points = exclude_points(main_building, outer_objects[0])
            if len(points) != 0:
                columns = detect_columns(points)
            else:
                columns = []
            floors = 1
            max_column = None
            for g in columns:
                if floors < len(g):
                    floors = len(g)
                    max_column = g

            if floors > 1 and main_building:
                y_distance = main_building[1] + main_building[3] / 2 - max_column[0][1] - max_column[0][3] / 2
                height_diff = [max_column[x][1] - max_column[x + 1][1] for x in range(floors - 1)]
                max_diff = np.max(height_diff)
                if y_distance / max_diff > 1.5:
                    floors += 1
            floors_list.append(str(floors) + "层")
        return floors_list

class AddedFloorPredictor(BasePredictor):
    class EfficientNetV2MultiLabel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.base_model = models.efficientnet_v2_l(weights=None)
            in_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features, num_classes),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.base_model(x)

    def __init__(self, weights_path: PathLike | str):
        super().__init__()
        self.__add_floor_model = AddedFloorPredictor.EfficientNetV2MultiLabel(1)
        self.__add_floor_model.load_state_dict(torch.load(weights_path))
        self.__add_floor_model.eval()
        self.__add_floor_model.to(self.device)
        self.__transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((448, 448)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, images=None):
        if images is None:
            images = self.last_image
        results = []
        for i in range(0, len(images), self.batch_size):
            img = images[i:i + self.batch_size]
            processed = self.transform_image(img, self.__transform)
            probs = self.__add_floor_model(processed).detach().cpu().numpy()
            result = map(lambda prob: '无加层' if prob > 0.5 else '有加层', probs)
            results.extend(list(result))
        return results

class MaterialPredictor(BasePredictor):
    class EfficientNetV2MultiLabel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.base = models.efficientnet_v2_l(weights=None)
            in_features = self.base.classifier[1].in_features
            self.base.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features, num_classes),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.base(x)

    def __init__(self, weights_path: PathLike | str):
        super().__init__()
        self.type_dict = np.asarray([f'类别{x},' for x in range(1, 13)])
        self.__material_model = self.EfficientNetV2MultiLabel(12)
        self.__material_model.load_state_dict(torch.load(weights_path))
        self.__material_model.eval()
        self.__material_model.to(self.device)

        self.__test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((448, 448)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, images=None):
        if images is None:
            images = self.last_image
        results = []
        for i in range(0, len(images), self.batch_size):
            img = images[i:i + self.batch_size]
            processed = self.transform_image(img, self.__test_transform)
            probs = self.__material_model(processed).detach().cpu().numpy()
            result = [''.join(self.type_dict[prob>0.5].tolist())[:-1] for prob in probs]
            results.extend(list(result))
        return results

class HiddenDangerPredictor(BasePredictor):
    type_names = {
        0: "空鼓",
        1: "渗水",
        2: "脱落",
        3: "裂缝"
    }  # 自定义类别名称

    @classmethod
    def calculate_polygon_area(cls, points):
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    @classmethod
    def process_results(cls, results):
        total_defects = 0
        total_area = 0
        str_results = []
        for i, result in enumerate(results):
            defects_info = []
            current_total = 0
            current_area = 0

            if hasattr(result, 'obb'):
                for j, obb in enumerate(result.obb):

                    points = obb.xyxyxyxy.reshape(4, 2).cpu().numpy()
                    danger_type = obb.cls.item()
                    area = cls.calculate_polygon_area(points)

                    name = cls.type_names.get(danger_type, "Unknown")

                    current_total += 1
                    current_area += area
                    defect_id = current_total

                    defects_info.append({
                        "id": defect_id,
                        "type": name,
                        "area": area
                    })
                str_results.append(json.dumps(defects_info))
                total_defects += current_total
                total_area += current_area
        return str_results, total_defects, total_area

    def __init__(self, weights_path: PathLike | str):
        super().__init__()
        self.__hidden_danger_model = YOLO(weights_path).eval()

    def predict(self, images=None):
        if images is None:
            images = self.last_image
        results = []
        for i in range(0, len(images), self.batch_size):
            danger_object = self.__hidden_danger_model.predict(
                source=images[i:i + self.batch_size],
                save=False,  # 关闭自动保存
                show=False,
                iou=0.1,
                conf=0.4,
                verbose=False
            )
            result, _, _ = self.process_results(danger_object)
            results.extend(result)

        return results