import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JD = (
    "Senior platform engineer with Kubernetes, AWS, Terraform, observability, "
    "and experience building reliable distributed systems."
)

CANDIDATES = [
    {
        "candidate_id": "CN-88210",
        "name": "Alex Strathmore",
        "role": "Principal Systems Architect",
        "summary": "12+ years of distributed systems experience at product companies building resilient cloud-native platforms.",
        "skills": ["Kubernetes", "AWS", "Terraform", "Go", "Rust"],
        "years_of_experience": 12,
        "response_rate": 0.94,
        "semantic_anchor": "platform engineering distributed systems",
        "product_fit": 0.95,
        "experience_score": 0.96,
        "behavior_score": 0.92,
    },
    {
        "candidate_id": "CN-90112",
        "name": "Sarah Jenkins",
        "role": "Senior Data Scientist",
        "summary": "Specialist in ML pipelines, embeddings, and evaluation frameworks for recommendation systems.",
        "skills": ["Python", "PyTorch", "Embeddings", "NLP"],
        "years_of_experience": 9,
        "response_rate": 0.91,
        "semantic_anchor": "machine learning evaluation embeddings",
        "product_fit": 0.91,
        "experience_score": 0.98,
        "behavior_score": 0.89,
    },
    {
        "candidate_id": "CN-77234",
        "name": "Marcus Thorne",
        "role": "ML Infrastructure Engineer",
        "summary": "Builds reliable training clusters, deployment automation, and production ML infrastructure.",
        "skills": ["Kubernetes", "MLflow", "Python", "AWS"],
        "years_of_experience": 8,
        "response_rate": 0.93,
        "semantic_anchor": "ml infrastructure cloud automation",
        "product_fit": 0.93,
        "experience_score": 0.95,
        "behavior_score": 0.95,
    },
    {
        "candidate_id": "CN-81203",
        "name": "Elena Rodriguez",
        "role": "Lead Frontend Engineer",
        "summary": "Leads high-scale frontend platforms and design systems for SaaS products.",
        "skills": ["React", "TypeScript", "Design Systems"],
        "years_of_experience": 10,
        "response_rate": 0.90,
        "semantic_anchor": "frontend design systems product delivery",
        "product_fit": 0.94,
        "experience_score": 0.90,
        "behavior_score": 0.94,
    },
    {
        "candidate_id": "CN-99012",
        "name": "Jordan Vane",
        "role": "DevOps Specialist",
        "summary": "Operates cloud platforms, CI/CD automation, and release engineering at scale.",
        "skills": ["AWS", "Jenkins", "Terraform", "Linux"],
        "years_of_experience": 7,
        "response_rate": 0.92,
        "semantic_anchor": "devops ci cd infrastructure",
        "product_fit": 0.88,
        "experience_score": 0.98,
        "behavior_score": 0.92,
    },
    {
        "candidate_id": "CN-44122",
        "name": "Tariq Mahmood",
        "role": "Backend Engineer (Go)",
        "summary": "Builds APIs, service-oriented systems, and reliability-focused backend platforms.",
        "skills": ["Go", "gRPC", "Kafka", "Postgres"],
        "years_of_experience": 6,
        "response_rate": 0.90,
        "semantic_anchor": "backend systems go microservices",
        "product_fit": 0.91,
        "experience_score": 0.92,
        "behavior_score": 0.90,
    },
    {
        "candidate_id": "CN-66509",
        "name": "Lydia Grant",
        "role": "Security Researcher",
        "summary": "Works on threat modeling, zero-trust architecture, and platform security reviews.",
        "skills": ["Security", "Cloud", "Threat Modeling"],
        "years_of_experience": 9,
        "response_rate": 0.96,
        "semantic_anchor": "security threat modeling platform hardening",
        "product_fit": 0.96,
        "experience_score": 0.88,
        "behavior_score": 0.97,
    },
]

STATE = {"latest_job": DEFAULT_JD, "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")}


def _text_overlap(text_a, text_b):
    words_a = set(text_a.lower().replace("-", " ").split())
    words_b = set(text_b.lower().replace("-", " ").split())
    if not words_a or not words_b:
        return 0.0
    overlap = words_a & words_b
    return len(overlap) / max(1, len(words_a | words_b))


def build_dataset(limit=100):
    rows = []
    for index, candidate in enumerate(CANDIDATES[:limit], start=1):
        rows.append(
            {
                "rank": index,
                "candidate_id": candidate["candidate_id"],
                "name": candidate["name"],
                "role": candidate["role"],
                "summary": candidate["summary"],
                "skills": candidate["skills"],
                "years_of_experience": candidate["years_of_experience"],
                "semantic_score": round(0.82 + (index * 0.015), 2),
                "fit_score": candidate["product_fit"],
                "experience_score": candidate["experience_score"],
                "behavior_score": candidate["behavior_score"],
                "composite_score": round(
                    (0.82 + (index * 0.015)) * 0.5 + candidate["product_fit"] * 0.2 + candidate["experience_score"] * 0.2 + candidate["behavior_score"] * 0.1,
                    3,
                ),
            }
        )
    return rows


def rank_candidates(jd_text, limit=10):
    jd_lower = (jd_text or DEFAULT_JD).lower()
    ranked = []
    for index, candidate in enumerate(CANDIDATES, start=1):
        semantic_overlap = _text_overlap(jd_lower, f"{candidate['semantic_anchor']} {candidate['role']} {' '.join(candidate['skills'])}")
        semantic_score = round(min(0.99, 0.7 + semantic_overlap * 0.25 + (candidate["years_of_experience"] % 5) * 0.01), 3)
        fit_score = candidate["product_fit"]
        experience_score = candidate["experience_score"]
        behavior_score = candidate["behavior_score"]
        composite_score = round((semantic_score * 0.5) + (fit_score * 0.2) + (experience_score * 0.2) + (behavior_score * 0.1), 3)
        reasoning = (
            f"Matched to {candidate['role']} with strong alignment on {', '.join(candidate['skills'][:3])}."
        )
        ranked.append(
            {
                "rank": 0,
                "candidate_id": candidate["candidate_id"],
                "name": candidate["name"],
                "role": candidate["role"],
                "summary": candidate["summary"],
                "skills": candidate["skills"],
                "years_of_experience": candidate["years_of_experience"],
                "score": composite_score,
                "semantic_score": semantic_score,
                "fit_score": fit_score,
                "experience_score": experience_score,
                "behavior_score": behavior_score,
                "confidence": round(composite_score * 100, 1),
                "reasoning": reasoning,
            }
        )
    ranked.sort(key=lambda item: (-item["score"], item["candidate_id"]))
    for idx, item in enumerate(ranked[:limit], start=1):
        item["rank"] = idx
    return ranked[:limit]


class HiringHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, relative_path):
        safe_path = os.path.normpath(relative_path)
        if safe_path.startswith("..") or safe_path.startswith("/"):
            self.send_error(404)
            return None
        file_path = os.path.join(BASE_DIR, safe_path)
        if os.path.isdir(file_path):
            file_path = os.path.join(file_path, "index.html")
        if os.path.exists(file_path) and os.path.isfile(file_path):
            with open(file_path, "rb") as handle:
                content = handle.read()
            self.send_response(200)
            self.send_header("Content-Type", self._content_type(file_path))
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return file_path
        self.send_error(404)
        return None

    def _content_type(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        return {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".json": "application/json; charset=utf-8",
        }.get(ext, "application/octet-stream")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/health":
            self._send_json({"status": "ok", "latest_job": STATE["latest_job"]})
            return
        if path == "/api/leaderboard":
            limit = int(parse_qs(parsed.query).get("limit", ["10"])[0])
            payload = {"job_description": STATE["latest_job"], "results": rank_candidates(STATE["latest_job"], limit=limit)}
            self._send_json(payload)
            return
        if path == "/api/dataset":
            limit = int(parse_qs(parsed.query).get("limit", ["100"])[0])
            self._send_json({"results": build_dataset(limit=limit)})
            return
        if path == "/":
            self._serve_file("index.html")
            return
        self._serve_file(path.lstrip("/"))

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/submit-job":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body or "{}")
            job_description = payload.get("job_description", DEFAULT_JD)
            STATE["latest_job"] = job_description
            STATE["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            results = rank_candidates(job_description, limit=8)
            self._send_json({"status": "ok", "job_description": job_description, "results": results})
            return
        self._send_json({"error": "Not found"}, 404)


def run_server(host="127.0.0.1", port=8000):
    server = ThreadingHTTPServer((host, port), HiringHandler)
    print(f"Hiring backend running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
