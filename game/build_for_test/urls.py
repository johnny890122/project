from django.contrib import admin
from django.urls import path, include, re_path
from django.views.generic import TemplateView
from django.http import JsonResponse
from rest_framework import status
from build.pages import Index
import re

urlpatterns = [
    path('123/', Index.vars_for_react),
    # path('/123', Index.vars_for_react),
]