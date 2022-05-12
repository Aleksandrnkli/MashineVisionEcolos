from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # There is no other applications at the moment of comment's creation,
    # so we use lpr_detection app as root endpoint
    path('admin/', admin.site.urls),
    path('api-auth/', include('rest_framework.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) \
  + [path('', include('lpr_detection.urls'))]
