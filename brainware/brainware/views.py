from django.shortcuts import render, redirect
from users.models import UserModel
from .utils import get_recommended_articles

from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods

import json

# Create your views here.

@login_required(redirect_field_name="login")
def index(request):
    fasttext_page = int(request.GET.get("fpage", 1))
    scibert_page = int(request.GET.get("spage", 1))

    user_model = UserModel.objects.get(user=request.user)

    interests = user_model.interests
    article_ids = user_model.get_articles_read()

    if user_model.interests == '':
        return redirect('profile')
    
    fasttext_articles, scibert_articles, total_pages = get_recommended_articles(interests, article_ids, (fasttext_page - 1, scibert_page - 1))
    
    return render(request, 'pages/index.html', {
        "fasttext_articles": fasttext_articles,
        "scibert_articles": scibert_articles,
        "fpage": fasttext_page,
        "spage": scibert_page,
        "total_pages": total_pages
    })

@require_http_methods(["POST"])
@login_required()
def read_article(request):
    user_model = UserModel.objects.get(user=request.user)
    
    data = json.loads(request.body)

    article_id = data["article_id"]

    user_model.add_article_read(article_id=article_id)
    user_model.save()

    return JsonResponse({'status': 'success'}, status=200)