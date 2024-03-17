# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path
from apps.home import views
from .views import ValidationView, Client1


urlpatterns = [

    # The home page
    path('ok', Client1.as_view(), name="clientv"),
    path('', views.index, name='home'),
    path('validation/', ValidationView.as_view(), name='validation_view'),
    # Matches any html file
    re_path(r'^.*\.*', views.pages, name='pages'),
    #path("", include("apps.home.urls"))

]
