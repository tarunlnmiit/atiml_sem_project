from django.urls import path
from . import views


urlpatterns = [
   path('', views.getQueries),
   path('getprediction', views.getPrediction),
   path('results', views.getResult)
]