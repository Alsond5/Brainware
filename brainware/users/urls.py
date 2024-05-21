from django.urls import path
from .views import *

urlpatterns = [
    path('register/', render_register, name='register'),
    path('login/', render_login, name='login'),
    path('logout/', render_logout, name='logout'),
    path('profile/', profile, name='profile'),
]
