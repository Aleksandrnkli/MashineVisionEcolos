# Generated by Django 3.1.6 on 2021-02-19 14:01

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import lpr_detection.models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='LPRDetection',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to=lpr_detection.models.UploadToPathAndRename('D:/detections/lpr_detections'))),
                ('detection_time', models.DateTimeField(auto_now_add=True, db_index=True)),
                ('license_plate', models.CharField(blank=True, db_index=True, default='', max_length=15)),
                ('marked_as_error', models.BooleanField(default=False)),
                ('creator', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='lpr_detection', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['detection_time'],
            },
        ),
    ]