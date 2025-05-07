from django.http import JsonResponse
from django.views import View

from .models import Task
from .predictors import FloorPredictor, AddedFloorPredictor, MaterialPredictor, HiddenDangerPredictor, BasePredictor
import json
import uuid

# Create your views here.

class CreateTaskView(View):
    def post(self, request):
        mac_str = bytes.fromhex(request.POST['mac'])
        task_list = request.POST['task_list']
        task_uuid = uuid.uuid1()
        new_task = Task(task_id=task_uuid.bytes, mac_address=mac_str, max_task_count=len(task_list.split(',')),
                        finished_task_count=0, tasks_list=task_list)
        new_task.save()
        print(mac_str)
        return JsonResponse({'task_id': task_uuid.hex, 'message':'success'}, status=200)


class FloorsPredictView(View):
    predictor = FloorPredictor('./main_building.pt', './outer_obj.pt')

    def post(self, request):
        # 检查是否有文件被上传
        res = self.predictor.get_image_from_request(request)
        if isinstance(res, JsonResponse):
            return res
        floors = self.predictor.predict()
        print("Floors:", floors)

        return JsonResponse({'message':'Upload success!', 'type':'floors', 'content':floors}, status=200)

class AddedFloorsPredictView(View):
    predictor = AddedFloorPredictor('./add_predict.pth')

    def post(self, request):
        res = self.predictor.get_image_from_request(request)
        if isinstance(res, JsonResponse):
            return res
        is_added = 0
        # is_added = self.predictor.predict()

        return JsonResponse({'message': 'Upload success!', 'type':'is_added', 'content': is_added}, status=200)

class MaterialPredictView(View):
    predictor = MaterialPredictor('./material.pth')
    def post(self, request):
        res = self.predictor.get_image_from_request(request)
        if isinstance(res, JsonResponse):
            return res
        material = "未知"
        # material = self.predictor.predict()
        # print("Materials:", material)
        return JsonResponse({'message': 'Upload success!', 'type':'material', 'content': material}, status=200)

class HiddenDangerPredictView(View):
    predictor = HiddenDangerPredictor('./best.pt')

    def post(self, request):
        res = self.predictor.get_image_from_request(request)
        if isinstance(res, JsonResponse):
            return res
        dangers = "未发现"
        return JsonResponse({'message': 'Not implement method.',
                             'type':'hidden_danger',
                             'content': dangers},
                            status=200)

class ComprehensivePredictView(View):
    predictors = {
        'floors': FloorPredictor('./main_building.pt', './outer_obj.pt'),
        'add': AddedFloorPredictor('./add_predict.pth'),
        'material': MaterialPredictor('./material.pth'),
        'hidden': HiddenDangerPredictor('./best.pt')
    }
    def post(self, request):
        options = json.loads(request.POST['options'])
        result = {}
        try:
            res, names = BasePredictor.get_image_from_request(request)
        except Exception as e:
            return JsonResponse({'message': 'Internal server error.', 'type':'error', 'content': str(e)}, status=500)
        if isinstance(res, JsonResponse):
            return res
        for option in options:
            if option in self.predictors:
                predictor = self.predictors[option]
                result[option] = predictor.predict(res)
        data = []
        for option in options:
            if option in self.predictors:
                for index, name in enumerate(names):
                    data.append({'name': name, 'result': result[option][index], 'type': option})

        return JsonResponse({'message': 'Upload success!', 'content': data}, status=200)