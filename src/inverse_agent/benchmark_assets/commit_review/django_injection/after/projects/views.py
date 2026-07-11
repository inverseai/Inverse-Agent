from django.db import connection
from django.http import JsonResponse


def search(request):
    query = request.GET.get("q", "")
    with connection.cursor() as cursor:
        cursor.execute(
            f"SELECT id, name FROM projects_project WHERE name LIKE '%{query}%'"
        )
        rows = [{"id": row[0], "name": row[1]} for row in cursor.fetchall()]
    return JsonResponse(rows, safe=False)
