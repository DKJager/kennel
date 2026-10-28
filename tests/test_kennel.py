import sys, os, json, pathlib, tempfile, shutil
# Insert the src directory onto the import path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

# Import public API
from kennel import KennelError, chat, ask_crowd, reconcile, ground, Votes, lambda_handler


import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import os, json, tempfile, threading
def test_client_posts_bearer_and_parses():
    """client.chat must send Bearer header and return assistant message."""
    captured = {}

    def dummy_opener(request, timeout):
        # Capture request details for assertions
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["auth"] = request.get_header("Authorization")
        captured["content_type"] = (request.get_header("Content-type") or request.get_header("Content-Type"))
        # Return a minimal OpenAI‑compatible response
        payload = {"choices": [{"message": {"content": "hello world"}}]}
        return json.dumps(payload).encode("utf-8")

    result = chat(
        base_url="https://example.com",
        api_key="test-token",
        model="model‑x",
        messages=[{"role": "user", "content": "hi"}],
        opener=dummy_opener,
    )
    assert captured["url"] == "https://example.com/chat/completions", "Incorrect endpoint URL"
    assert captured["method"] == "POST", "Request method should be POST"
    assert captured["auth"] == "Bearer test-token", "Missing or wrong Authorization header"
    assert captured["content_type"] == "application/json", "Content‑Type must be application/json"
    assert result == "hello world", "Returned assistant content does not match"


def test_crowd_survives_failed_model():
    """ask_crowd must continue even if one model fails."""
    def ask_fn(model, question):
        if model == "bad-model":
            raise RuntimeError("failure")
        return f"answer‑from‑{model}"

    results = ask_crowd("question?", ["good-model", "bad-model"], ask_fn)
    assert isinstance(results, list) and len(results) == 2, "Expected two result entries"
    # Identify good and bad entries
    good = next(r for r in results if r["model"] == "good-model")
    bad = next(r for r in results if r["model"] == "bad-model")
    assert good["error"] is None, "Good model should have no error"
    assert good["answer"] == "answer‑from‑good-model", "Good model answer mismatch"
    assert bad["error"] is not None, "Failed model must report an error"
    assert bad["answer"] == "", "Failed model answer should be empty string"


def test_reconcile_agreement_dissent():
    """reconcile must separate agreement, dissent and compute confidence."""
    answers = [
        {"model": "a", "answer": "The cat sits. The dog barks.", "error": None},
        {"model": "b", "answer": "The cat sits. The bird flies.", "error": None},
    ]
    out = reconcile(answers)
    assert isinstance(out, dict), "reconcile should return a dict"
    agreement = out.get("agreement")
    dissent = out.get("dissent")
    confidence = out.get("confidence")
    assert agreement == ["the cat sits"], "Agreement should contain the normalized shared claim"
    assert set(dissent) == {"the dog barks", "the bird flies"}, "Dissent should list the unique claims"
    assert confidence == 1.0, "Confidence should be 1.0 when all models agree on at least one claim"


def test_ground_marks_ungrounded():
    """ground must mark claims with low overlap as ungrounded."""
    claims = ["alpha claim", "beta claim"]
    passages = [
        {"id": "p1", "text": "this passage mentions alpha claim somewhere"},
        {"id": "p2", "text": "completely unrelated text"},
    ]
    grounded = ground(claims, passages)
    assert isinstance(grounded, list) and len(grounded) == 2, "Expected two grounded entries"
    alpha = next(item for item in grounded if item["claim"] == "alpha claim")
    beta = next(item for item in grounded if item["claim"] == "beta claim")
    assert alpha["source_id"] == "p1", "Alpha claim should be linked to passage p1"
    assert alpha["score"] >= 0.2, "Alpha score should be >= 0.2"
    assert beta["source_id"] is None, "Beta claim should be ungrounded (source_id None)"
    assert beta["score"] < 0.2, "Beta score should be below 0.2"


def test_votes_weights_smoothing():
    """Votes.weights must apply Laplace smoothing."""
    tmp_dir = tempfile.mkdtemp()
    try:
        store = Votes(tmp_dir)
        # Record three votes for model 'a' (2 helpful, 1 unhelpful)
        store.record("q1", "a", True)
        store.record("q2", "a", True)
        store.record("q3", "a", False)
        # No votes for 'b' and 'c'
        weights = store.weights(["a", "b", "c"])
        assert isinstance(weights, dict), "weights must return a dict"
        assert abs(weights["a"] - 0.6) < 1e-9, "Model a weight should be (2+1)/(3+2)=0.6"
        assert abs(weights["b"] - 0.5) < 1e-9, "Model b (no votes) should default to 0.5"
        assert abs(weights["c"] - 0.5) < 1e-9, "Model c (no votes) should default to 0.5"
    finally:
        shutil.rmtree(tmp_dir)


def test_handler_no_key_503():
    """lambda_handler should return 503 when API key missing and no ask_fn."""
    # Ensure the environment variable is unset for this test
    prev = os.environ.pop("KENNEL_API_KEY", None)
    try:
        event = {
            "httpMethod": "POST",
            "path": "/",
            "body": json.dumps({"question": "test?"}),
        }
        resp = lambda_handler(event, None)
        assert resp["statusCode"] == 503, "Expected 503 when API key is missing"
        assert "Access-Control-Allow-Origin" in resp.get("headers", {}), "CORS header missing"
    finally:
        if prev is not None:
            os.environ["KENNEL_API_KEY"] = prev


def test_handler_post_flow():
    """Full POST flow should produce answers, reconciliation, grounding and weights."""
    os.environ["KENNEL_API_KEY"] = "dummy-key"
    os.environ["KENNEL_MODELS"] = "model1,model2"
    os.environ["KENNEL_BASE_URL"] = "http://dummy.base"

    # Simple ask_fn that returns a deterministic answer per model
    def fake_ask(model, question):
        return f"{question} answered by {model}"

    event = {
        "httpMethod": "POST",
        "path": "/",
        "body": json.dumps({"question": "What is AI?"}),
    }
    resp = lambda_handler(event, None, ask_fn=fake_ask)
    assert resp["statusCode"] == 200, "Expected 200 OK for successful POST"
    body = json.loads(resp["body"])
    answers = body.get("answers")
    assert isinstance(answers, list) and len(answers) == 2, "Should return two model answers"
    models_returned = {a["model"] for a in answers}
    assert models_returned == {"model1", "model2"}, "Returned models mismatch"
    # Check that each answer contains the injected text
    for a in answers:
        assert "answered by" in a["answer"], "Answer content missing expected pattern"
    # Reconcile must be present
    assert "reconcile" in body, "Missing reconcile section"
    # Grounded section must be present (may be empty)
    assert "grounded" in body, "Missing grounded section"
    # Weights must be present and include both models
    weights = body.get("weights", {})
    assert set(weights.keys()) == {"model1", "model2"}, "Weights missing for models"


def test_cors_and_validation():
    """All responses must have CORS headers and input validation must reject bad data."""
    # GET request should include CORS
    get_event = {"httpMethod": "GET", "path": "/"}
    resp = lambda_handler(get_event, None)
    assert resp["statusCode"] == 200, "GET should succeed"
    for hdr in ("Access-Control-Allow-Origin", "Access-Control-Allow-Methods", "Access-Control-Allow-Headers"):
        assert hdr in resp.get("headers", {}), f"CORS header {hdr} missing in GET response"

    # POST with empty question should yield 400
    os.environ["KENNEL_API_KEY"] = "dummy-key"
    bad_event = {
        "httpMethod": "POST",
        "path": "/",
        "body": json.dumps({"question": ""}),
    }
    bad_resp = lambda_handler(bad_event, None)
    assert bad_resp["statusCode"] == 400, "Empty question should trigger 400 Bad Request"
    for hdr in ("Access-Control-Allow-Origin",):
        assert hdr in bad_resp.get("headers", {}), "CORS header missing in error response"


def test_reconcile_empty_safe():
    """reconcile must handle empty input without raising and return safe defaults."""
    out = reconcile([])
    assert out["agreement"] == [], "Agreement list should be empty for no answers"
    assert out["dissent"] == [], "Dissent list should be empty for no answers"
    assert out["confidence"] == 0.0, "Confidence should be 0.0 when no answers are present"


def test_default_models_include_nemotron():
    """When KENNEL_MODELS is unset, default list must contain the Nemotron model."""
    # Remove any user‑provided model list
    os.environ.pop("KENNEL_MODELS", None)
    # Use a minimal ask_fn to avoid network calls
    def stub_ask(model, q):
        return "stub"

    event = {
        "httpMethod": "POST",
        "path": "/",
        "body": json.dumps({"question": "test"}),
    }
    resp = lambda_handler(event, None, ask_fn=stub_ask)
    body = json.loads(resp["body"])
    models_used = {a["model"] for a in body.get("answers", [])}
    assert any("nvidia/" in m and "nemotron" in m.lower() for m in models_used) and any(not m.startswith("nvidia/") for m in models_used), "Default models must include Nemotron"


def test_readme_and_license_files():
    """LICENSE must contain Apache text, NOTICE the NVIDIA line, README the required lines and no '/opt'."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]

    license_path = repo_root / "LICENSE"
    notice_path = repo_root / "NOTICE"
    readme_path = repo_root / "README.md"

    assert license_path.is_file(), "LICENSE file missing"
    lic_content = license_path.read_text(encoding="utf-8")
    assert "Apache License" in lic_content, "LICENSE must contain Apache License text"

    assert notice_path.is_file(), "NOTICE file missing"
    notice_content = notice_path.read_text(encoding="utf-8").strip()
    expected_notice = "Licensed by NVIDIA Corporation under the NVIDIA Open Model License."
    assert notice_content == expected_notice, "NOTICE must contain the exact NVIDIA line"

    assert readme_path.is_file(), "README.md file missing"
    readme_content = readme_path.read_text(encoding="utf-8")
    assert expected_notice in readme_content, "README must contain the NVIDIA line"
    assert "Builder credits used: Nebius Token Factory" in readme_content, "README missing Builder credits line"
    import re as _re
    assert _re.search(r"First commit \(UTC\): \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", readme_content), "README missing first-commit timestamp"
    assert "/opt" not in readme_content, "README must not contain any '/opt' paths"


if __name__ == "__main__":
    # Run each test function in order
    i_test_client_posts_bearer_and_parses()
    ii_test_crowd_survives_failed_model()
    iii_test_reconcile_agreement_dissent()
    iv_test_ground_marks_ungrounded()
    v_test_votes_weights_smoothing()
    vi_test_handler_no_key_503()
    vii_test_handler_post_flow()
    viii_test_cors_and_validation()
    ix_test_reconcile_empty_safe()
    x_test_default_models_include_nemotron()
    xi_test_readme_and_license_files()
    print("PASS")

def test_client_sends_max_tokens_and_temperature():
    captured = {}
    def fake_opener(request, timeout):
        captured['data'] = json.loads(request.data.decode())
        return json.dumps({
            "choices": [{"message": {"content": "ok"}}]
        }).encode()
    chat(
        base_url="https://api.test/v1",
        api_key="KEY",
        model="m",
        messages=[],
        max_tokens=123,
        temperature=0.7,
        opener=fake_opener,
    )
    data = captured['data']
    assert data.get("max_tokens") == 123, "max_tokens not sent"
    assert abs(data.get("temperature", 0) - 0.7) < 1e-9, "temperature not sent"

# (iii) ask_crowd survives a failing model
def test_handler_sends_system_prompt_and_limits(monkeypatch):
    import kennel.client as kc
    calls = []
    def fake_chat(base, key, model, messages, timeout=60, opener=None, max_tokens=None, temperature=None):
        calls.append({"messages": messages, "timeout": timeout, "max_tokens": max_tokens, "temperature": temperature})
        return "Plain answer sentence one. Plain answer sentence two."
    monkeypatch.setattr(kc, "chat", fake_chat)
    monkeypatch.setenv("KENNEL_API_KEY", "K")
    monkeypatch.setenv("KENNEL_MODELS", "m1,m2")
    event = {"httpMethod": "POST", "path": "/", "body": json.dumps({"question": "Test?", "passages": []})}
    resp = lambda_handler(event, None)
    assert resp["statusCode"] == 200
    assert len(calls) == 2
    for c in calls:
        assert c["messages"][0]["role"] == "system" and "3 to 5 short" in c["messages"][0]["content"]
        assert c["messages"][1] == {"role": "user", "content": "Test?"}
        assert c["max_tokens"] == 300 and abs(c["temperature"] - 0.2) < 1e-9 and c["timeout"] == 40

def test_reconcile_strips_markdown_and_abbrev():
    answers = [
        {"model":"m1","answer":"**Coffee** is fine for most adults.", "error":None},
        {"model":"m2","answer":"Coffee is fine for most adults.", "error":None},
        {"model":"m3","answer":"e.g. example.", "error":None},
    ]
    res = reconcile(answers)
    # "coffee is fine for most adults" should appear once in agreement (since >=2 models)
    assert any("coffee is fine for most adults" == c for c in res["agreement"]), "markdown not stripped"
    # "e.g." should not become a claim "g"
    assert not any(c == "g" for c in res["agreement"] + res["dissent"]), "abbrev produced bad claim"

# ----------------------------------------------------------------------
if __name__ == "__main__":
    test_client_posts_bearer_and_parses()
    test_client_sends_max_tokens_and_temperature()
    test_crowd_survives_failed_model()
    test_reconcile_agreement_dissent()
    test_reconcile_empty_safe()
    test_ground_marks_ungrounded()
    test_ground_stopword_overlap()
    test_votes_weights_smoothing()
    test_handler_no_key_503()
    test_handler_post_flow()
    test_cors_and_validation()
    test_default_models_include_nemotron()
    test_readme_and_license_files()
    test_handler_sends_system_prompt_and_limits()
    test_reconcile_strips_markdown_and_abbrev()
    print("ALL TESTS PASSED")
def test_ground_stopword_overlap():
    claim = "moderate coffee intake lowers cardiovascular risk"
    passage = {"id": "s1", "text": "Coffee intake can lower the risk of heart disease."}
    result = ground([claim], [passage])[0]
    assert result["source_id"] == "s1", "grounding failed"
    assert result["score"] >= 0.2, "score below threshold"

# (viii) Votes weights smoothing works

def test_dash_separates_words():
    out = reconcile([{"model": "a", "answer": "Moderate intake—about three cups daily—is fine for adults.", "error": None},
                     {"model": "b", "answer": "Moderate intake—about three cups daily—is fine for adults.", "error": None}])
    assert out["agreement"] and "intakeabout" not in out["agreement"][0] and "intake about" in out["agreement"][0]


def test_single_shared_word_does_not_ground():
    g = ground(["heart failure risk rises", "moderate coffee intake lowers cardiovascular risk"],
               [{"id": "s1", "text": "Moderate coffee intake of 3 to 5 cups a day is associated with lower cardiovascular risk in large cohort studies."}])
    assert g[0]["source_id"] is None
    assert g[1]["source_id"] == "s1" and g[1]["score"] >= 0.5


def test_get_root_serves_html():
    r = lambda_handler({"httpMethod": "GET", "path": "/"}, None)
    assert r["statusCode"] == 200 and r["headers"]["Content-Type"].startswith("text/html")
