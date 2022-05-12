from django.urls import path, include, re_path

from rest_framework.routers import DefaultRouter
from rest_framework.authtoken.views import obtain_auth_token

from lpr_detection import views


router = DefaultRouter()
router.register(r'lpr', views.LPRDetectionViewSet)
router.register(r'users', views.UserViewSet)

urlpatterns = [
    path('api/', include(router.urls)),
    path('api_token_auth/', obtain_auth_token, name='api_token_auth'),
    re_path(r'.*', views.index, name='index'),
]
