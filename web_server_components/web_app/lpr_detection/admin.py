from django.contrib import admin

from lpr_detection.models import LPRDetection

# Delete when deploying in production
# since in my opinion adding through admin panel is illegal
# and can cause collisions on pictures of cars (incorrect filenames)
admin.site.register(LPRDetection)
