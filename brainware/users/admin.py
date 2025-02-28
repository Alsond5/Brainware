from django.contrib import admin
from .models import UserModel

# Register your models here.

class UserModelAdmin(admin.ModelAdmin):
    list_display = ('user', 'interests')

    # list_filter = ("status",)
    # search_fields = ['title', 'content']
    # prepopulated_fields = {'slug': ('title',)}

admin.site.register(UserModel, UserModelAdmin)