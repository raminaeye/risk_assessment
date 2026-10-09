"""WSGI entry point: exposes the Flask app for production servers (e.g. gunicorn)."""
from app import app


if __name__ == "__main__":
    app.run()
