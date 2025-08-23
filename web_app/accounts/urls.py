from django.urls import path
from .views import SignInView, SignOutView, register

app_name = "accounts"

urlpatterns = [
    path("login/", SignInView.as_view(), name="login"),
    path("logout/", SignOutView.as_view(), name="logout"),
    path("register/", register, name="register"),
]
