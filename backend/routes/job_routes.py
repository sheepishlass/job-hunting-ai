import os
import time
from flask import Blueprint, jsonify, request, render_template
from backend.services.adzuna_service import AdzunaService
from backend.services.ml_service import MLService
from backend.services.matching_service import MatchingService
from backend.utils.validators import validate_search_request
import logging

logger = logging.getLogger(__name__)

jobs_bp = Blueprint("jobs_bp", __name__, url_prefix="/search")

# Initialize services (in production, replace with DI)
ml_service = MLService()
matching_service = MatchingService(ml_service)
adzuna_service = AdzunaService(
    app_id=os.getenv("ADZUNA_APP_ID"), app_key=os.getenv("ADZUNA_APP_KEY")
)

def adjust_experience_field(payload: dict) -> dict:
    """
    Adjust the 'experience' field in the request payload.
    Converts numeric experience (e.g., 2) into a level string
    ('entry', 'mid', 'senior') so that validation passes.
    """
    data = dict(payload)
    exp = data.get("experience")

    # If already a valid text level, leave it alone
    if isinstance(exp, str) and exp.lower() in ["entry", "mid", "senior"]:
        return data

    # Convert numeric or numeric-string experience to level
    if isinstance(exp, (int, float)) or (isinstance(exp, str) and exp.isdigit()):
        years = int(exp)
        if years <= 1:
            level = "entry"
        elif years <= 4:
            level = "mid"
        else:
            level = "senior"
        data["experience"] = level

    return data


@jobs_bp.route("/")
def search():
    return render_template("search.html", page_name="Job Search")


@jobs_bp.route("/results", methods=["POST"])
def search_jobs():
    start_time = time.time()

    data = request.get_json() or {}

    # Use a *copy* of the payload with adjusted experience ONLY for validation
    validation_payload = adjust_experience_field(data)

    # Validate request using the adjusted copy
    is_valid, error = validate_search_request(validation_payload)
    if not is_valid:
        return jsonify({"success": False, "error": error}), 400

    # Fetch jobs from Adzuna
    jobs = adzuna_service.search_jobs(
        keywords=data.get("keywords", None),
        location=data.get("location", None),
        distance=data.get("distance", None),
        max_results=data.get("max_results", 50),
    )

    if not jobs:
        return jsonify(
            {
                "success": True,
                "count": 0,
                "results": [],
                "message": "No jobs found matching criteria",
            }
        )

    # Rank jobs using ML
    matched_jobs = matching_service.rank_jobs(
        user_data=data, jobs=jobs, top_k=data.get("max_results", 20)
    )

    # Format response
    query_time = int((time.time() - start_time) * 1000)

    results = {
            "success": True,
            "count": len(matched_jobs),
            "query_time_ms": query_time,
            "results": [job.to_dict() for job in matched_jobs]
    }

    logger.info("search_ok", extra={"ctx": {"count": results["count"], "query_time_ms": query_time}})

    return render_template("results.html", page_name="Job Results", **results)
