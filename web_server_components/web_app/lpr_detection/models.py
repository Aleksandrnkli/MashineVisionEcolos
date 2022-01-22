from django.db import models
from django.utils.deconstruct import deconstructible
import uuid
import os


@deconstructible
class UploadToPathAndRename(object):
    """
    Auxiliary class for ImageField in LPRDetection model. Provides custom logic
    of saving image with UUID name
    """

    def __init__(self, path):
        self.sub_path = path
        if not os.path.exists(self.sub_path):
            os.makedirs(self.sub_path)

    def __call__(self, instance, filename):
        filename = f'{uuid.uuid4()}.jpg'

        return os.path.join(self.sub_path, filename)


class LPRDetection(models.Model):
    image = models.ImageField(upload_to=UploadToPathAndRename('lpr_detections'))
    detection_time = models.DateTimeField(db_index=True, auto_now_add=True)
    license_plate = models.CharField(max_length=15, default='', db_index=True)
    marked_as_error = models.BooleanField(default=False)
    creator = models.ForeignKey('auth.User', related_name='lpr_detection', on_delete=models.CASCADE)

    class Meta:
        ordering = ['detection_time']
