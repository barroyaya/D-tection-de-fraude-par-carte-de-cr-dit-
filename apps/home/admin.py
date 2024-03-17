# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
from .models import Banque
from django.contrib import admin
@admin.register(Banque)
class BanqueAdmin(admin.ModelAdmin):
    pass

# Register your models here.
