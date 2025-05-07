from django.urls import path

from . import views

urlpatterns = [
    path('task/', views.CreateTaskView.as_view(), name='task'),
    path('floors/', views.FloorsPredictView.as_view(), name='floors'),
    path('add/', views.AddedFloorsPredictView.as_view(), name='add'),
    path('material/', views.MaterialPredictView.as_view(), name='material'),
    path('hidden/', views.HiddenDangerPredictView.as_view(), name='hidden'),
    path('', views.ComprehensivePredictView.as_view(), name='comprehensive'),
]