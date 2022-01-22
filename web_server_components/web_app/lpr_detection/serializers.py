from rest_framework import serializers
from lpr_detection.models import LPRDetection
from django.contrib.auth.models import User


class UserSerializer(serializers.HyperlinkedModelSerializer):
    lpr_detection = serializers.HyperlinkedRelatedField(many=True, view_name='lprdetection-detail', read_only=True)

    class Meta:
        model = User
        fields = ['url', 'id', 'username', 'lpr_detection']


class LPRDetectionSerializer(serializers.HyperlinkedModelSerializer):
    creator = serializers.ReadOnlyField(source='creator.username')

    class Meta:
        model = LPRDetection
        fields = ['id', 'url', 'detection_time', 'license_plate', 'marked_as_error', 'creator', 'image']
