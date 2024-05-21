from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout 
from django.contrib.auth.models import User

from django.core.handlers.wsgi import WSGIRequest
from django.contrib.auth.decorators import login_required

from .models import UserModel

# Create your views here.

def is_authenticated_user(view_func):
    def wrapper(request: WSGIRequest, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect('index')
        
        return view_func(request, *args, **kwargs)
    
    return wrapper

def sign_in(request: WSGIRequest):
    if all(key in request.POST for key in ['username', 'password']) is False:
        return render(request, 'users/login.html', {'error': 'Geçersiz parametre girildi', 'username': ''})

    username = request.POST['username']
    password = request.POST['password']

    if username == '' or password == '':
        return render(request, 'users/login.html', {'error': 'Geçersiz kullanıcı adı veya şifre', 'username': username})

    user = authenticate(request, username=username, password=password)

    if user is None:
        return render(request, 'users/login.html', {'error': 'Geçersiz kullanıcı adı veya şifre', 'username': username})
    
    login(request, user)

    return redirect('index')

def sign_up(request: WSGIRequest):
    if all(key in request.POST for key in ['username', 'password', 'confirm-password', 'first-name', 'surname']) is False:
        return render(request, 'users/register.html')
    
    username = request.POST['username']
    password = request.POST['password']
    first_name = request.POST['first-name']
    surname = request.POST['surname']
    repassword = request.POST['confirm-password']

    if username == '' or password == '' or first_name == '' or surname == '' or repassword == '':
        return render(request, 'users/register.html')

    if password != repassword:
        return render(request, 'users/register.html', {'error': 'Passwords do not match'})
    
    if User.objects.filter(username=username).exists():
        return render(request, 'users/register.html', {'error': 'Username already exists'})
    
    user = User.objects.create_user(username=username, password=password, first_name=first_name, last_name=surname)
    user.save()

    user_model = UserModel.objects.create(user=user)
    user_model.save()

    return redirect('login')

@is_authenticated_user
def render_register(request: WSGIRequest):
    if request.method == 'POST':
        return sign_up(request)

    return render(request, 'users/register.html')

@is_authenticated_user
def render_login(request: WSGIRequest):
    if request.method == 'POST':
        return sign_in(request)

    return render(request, 'users/login.html')

def recover_account(request: WSGIRequest):
    return render(request, 'users/recover-account.html')

@login_required
def render_logout(request: WSGIRequest):
    logout(request)

    return redirect('login')

@login_required
def profile(request: WSGIRequest):
    user = request.user
    user_model = UserModel.objects.get(user=user)

    if request.method == 'POST':
        if all(key in request.POST for key in ['grid-interests', 'grid-first-name', 'grid-last-name']) is False:
            return render(request, 'users/profile.html', {'user_model': user_model})
        
        user_model.interests = request.POST['grid-interests']
        
        first_name = request.POST['grid-first-name']
        last_name = request.POST['grid-last-name']

        if first_name != user.first_name:
            user.first_name = first_name

        if last_name != user.last_name:
            user.last_name = last_name

        password = request.POST.get("password")

        if password and password != '':
            user.set_password(password)

        user.save()
        user_model.save()
        
        return redirect("index")
    
    return render(request, 'users/profile.html', {'user_model': user_model})