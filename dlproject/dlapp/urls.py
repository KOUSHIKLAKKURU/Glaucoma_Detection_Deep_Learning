from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns=[
    path('',views.homepage,name='home'),
    path('predict',views.predict,name='predict'),
    path('result',views.result,name='result'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)