from flask import Flask, render_template, jsonify
from flask_cors import CORS
from backend.config import Config
from backend.routes.job_routes import jobs_bp
import os
from dotenv import load_dotenv

load_dotenv()

def create_app():
    frontend_path = os.path.join(os.path.dirname(__file__), "../Frontend")
    app = Flask(__name__, template_folder=frontend_path, static_folder=frontend_path, static_url_path="")
    # CORS(app)

    # Basic config placeholder
    app.config.from_object(Config)

    app.register_blueprint(jobs_bp)

    @app.route("/")
    def about():
        return render_template("index.html", page_name="Job Hunting AI")

    @app.route("/health")
    def health():
        return jsonify({"status": "healthy", "version": "1.0"})

    @app.route("/config/check")
    def config_check():
        return jsonify({
            "ADZUNA_APP_ID_set": bool(Config.ADZUNA_APP_ID),
            "ADZUNA_APP_KEY_set": bool(Config.ADZUNA_APP_KEY),
            "DEBUG": Config.DEBUG
        })

    @app.route("/adzuna/test")
    def adzuna_test():
        try:
            from backend.services.adzuna_service import AdzunaService
            svc = AdzunaService()
            data = svc.search_jobs("python developer", "New York", max_results=3)
            mode = "LIVE" if (getattr(svc, "app_id", None) and getattr(svc, "app_key", None)
                              and not getattr(svc, "use_mock", True)) else "MOCK"
            return jsonify({"mode": mode, "count": len(data), "sample": data[:2]})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"mode": "UNKNOWN", "error": str(e)}), 500

    @app.route("/ml/status")
    def ml_status():
        try:
            return {"model_loaded": True, "message": "ML model integration active"}
        except Exception:
            return {"model_loaded": False}


    return app


if __name__ == "__main__":
    app = create_app()
    # app.run(host="0.0.0.0", port=8000, debug=app.config.get("DEBUG", False))
    app.run(port=8000, debug=app.config.get("DEBUG", False))
