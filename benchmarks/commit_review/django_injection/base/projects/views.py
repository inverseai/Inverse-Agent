from django.http import JsonResponse

from .models import Project


def search(request):
    query = request.GET.get("q", "")
    rows = Project.objects.filter(name__icontains=query).values("id", "name")
    return JsonResponse(list(rows), safe=False)
