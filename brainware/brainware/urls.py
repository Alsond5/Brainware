from django.urls import path
from .views import *

urlpatterns = [
    path('', index, name='index'),
    path('read-article', read_article, name='read-article'),
]