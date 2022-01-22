from django.contrib.auth.models import User
from django.shortcuts import render

from rest_framework import viewsets

from django_filters import rest_framework as filters

from lpr_detection.models import LPRDetection
from lpr_detection.serializers import LPRDetectionSerializer, UserSerializer


def index(request):
    """
    Plain load of Vue SPA app
    """
    return render(request, 'lpr_detection/index.html')


class UserViewSet(viewsets.ReadOnlyModelViewSet):
    """
    This viewset automatically provides 'list' and 'retrieve' actions.
    """
    queryset = User.objects.all()
    serializer_class = UserSerializer

class LPRDetectionFilter(filters.FilterSet):
    time_range = filters.DateTimeFromToRangeFilter(field_name='detection_time')

    class Meta:
        model = LPRDetection
        fields = ('detection_time', 'license_plate')


class LPRDetectionViewSet(viewsets.ModelViewSet):
    """
    This viewset automatically provides 'list', 'create', 'retrieve',
    'update' and 'destroy' actions.
    """
    queryset = LPRDetection.objects.all()
    serializer_class = LPRDetectionSerializer
    filter_backends = (filters.DjangoFilterBackend,)
    filterset_class = LPRDetectionFilter

    def perform_create(self, serializer):
        serializer.save(creator=self.request.user)
