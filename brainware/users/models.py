from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class UserModel(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    interests = models.TextField(blank=True, default='')
    articles_read = models.TextField(blank=True, default='')
    articles_liked = models.TextField(blank=True, default='')

    def __str__(self):
        return self.user.username
    
    def get_articles_read(self):
        if self.articles_read != "":
            return [int(article_id) for article_id in self.articles_read.split(',')]
    
        return []

    def get_articles_liked(self):
        return self.articles_liked.split(',')
    
    def get_interests(self):
        return self.interests.split(',')
    
    def add_article_read(self, article_id):
        articles = self.get_articles_read()
        articles.append(article_id)

        self.articles_read = ','.join([str(article_id) for article_id in articles])

        self.save()

    def add_article_liked(self, article_id):
        articles = self.get_articles_liked()
        articles.append(str(article_id))

        self.articles_liked = ','.join(articles)

        self.save()

    def add_interest(self, interest):
        interests = self.get_interests()
        interests.append(interest)

        self.interests = ','.join(interests)

        self.save()

    def remove_article_liked(self, article_id):
        articles = self.get_articles_liked()
        articles.remove(article_id)

        self.articles_liked = ','.join(articles)

        self.save()