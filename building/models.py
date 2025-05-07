from datetime import datetime

from django.db import models

# Create your models here.
class Task(models.Model):
    task_id = models.BinaryField(primary_key=True)
    max_task_count = models.IntegerField()
    finished_task_count = models.IntegerField()
    mac_address = models.BinaryField()
    tasks_list = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return str(self.task_id)
