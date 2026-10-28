import json
import os
from typing import Any, Dict, Optional

DEFAULT_MODELS = "nvidia/nemotron-3-super-120b-a12b,Qwen/Qwen3-30B-A3B-Instruct-2507,google/gemma-3-27b-it"
DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}

HTML_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KENNEL · Crowd Answers</title>
<style>
:root{--bg:#f6f5f1;--card:#fff;--ink:#1d1f1e;--mut:#6a6f6b;--line:#e3e1da;--acc:#76b900;--acc2:#2f6f3e;--warn:#b5651d}
@media (prefers-color-scheme:dark){:root{--bg:#121413;--card:#1b1e1c;--ink:#ecefe9;--mut:#9aa39c;--line:#2c312d;--acc:#8fd400;--acc2:#7fc48f;--warn:#e0a060}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
.wrap{max-width:860px;margin:0 auto;padding:28px 16px 60px}
header h1{margin:0;font-size:28px;letter-spacing:.04em}header p{margin:6px 0 0;color:var(--mut)}
.box{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-top:18px}
label{font-weight:600;font-size:14px}textarea{width:100%;margin-top:6px;padding:10px;border:1px solid var(--line);border-radius:8px;background:transparent;color:var(--ink);font:inherit;resize:vertical}
#q{min-height:70px}#src{min-height:60px;font-size:14px}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:12px}
button{border:0;border-radius:8px;padding:10px 18px;font:600 15px system-ui;cursor:pointer;background:var(--acc);color:#0b0d0c}
button.ghost{background:transparent;border:1px solid var(--line);color:var(--ink);font-weight:500;padding:6px 10px;font-size:13px}
button:disabled{opacity:.5;cursor:wait}.hint{color:var(--mut);font-size:13px}
.verdict h2{margin:0 0 8px;font-size:15px;text-transform:uppercase;letter-spacing:.08em;color:var(--mut)}
.meter{height:10px;background:var(--line);border-radius:6px;overflow:hidden;margin:6px 0 4px}.meter i{display:block;height:100%;background:var(--acc)}
ul{margin:6px 0 0;padding-left:20px}li{margin:4px 0}
.tag{display:inline-block;font-size:12px;padding:1px 7px;border-radius:10px;border:1px solid var(--line);color:var(--mut);margin-left:6px}
.tag.ok{border-color:var(--acc2);color:var(--acc2)}.tag.no{border-color:var(--warn);color:var(--warn)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin-top:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.card h3{margin:0 0 6px;font-size:14px}.card .maker{color:var(--mut);font-size:12px}
.card p{margin:8px 0;font-size:14px}.err{color:var(--warn);font-size:13px}
.votes{display:flex;gap:6px;align-items:center}.w{font-size:12px;color:var(--mut)}
footer{margin-top:30px;color:var(--mut);font-size:12px}
</style>
</head>
<body>
<div class="wrap">
<header>
<h1>KENNEL</h1>
<p>Ask once. Several open models from different makers answer independently. KENNEL shows where they agree, where they disagree, and which claims your sources back up.</p>
</header>

<div class="box">
<label for="q">Your question</label>
<textarea id="q" maxlength="4000" placeholder="e.g. Is coffee bad for your heart?"></textarea>
<label for="src" style="display:block;margin-top:12px">Sources to check against <span class="hint">(optional, one passage per line)</span></label>
<textarea id="src" placeholder="Paste a passage from a study, policy or document. Each line is one source."></textarea>
<div class="row"><button id="ask">Ask the crowd</button><span id="status" class="hint"></span></div>
</div>

<div id="out" hidden>
<div class="box verdict">
<h2>Where the models agree</h2>
<div class="meter"><i id="bar" style="width:0"></i></div>
<div class="hint" id="conf"></div>
<ul id="agree"></ul>
<h2 style="margin-top:16px">Said by only one model</h2>
<ul id="dissent"></ul>
</div>
<div class="grid" id="cards"></div>
</div>

<footer>Runs on Nebius Token Factory with NVIDIA Nemotron, Qwen and Gemma open models. Open source under Apache-2.0. Licensed by NVIDIA Corporation under the NVIDIA Open Model License.</footer>
</div>
<script>
const $=id=>document.getElementById(id);
const MAKERS={"nvidia":"NVIDIA","Qwen":"Alibaba","google":"Google","meta-llama":"Meta","deepseek-ai":"DeepSeek"};
let qid="";
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function srcTag(claim,grounded){const g=(grounded||[]).find(x=>x.claim===claim);if(!g)return"";return g.source_id?`<span class="tag ok">source ${esc(g.source_id)}</span>`:`<span class="tag no">no source</span>`}
async function vote(model,helpful,btn){
  try{const r=await fetch("/vote",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question_id:qid,model,helpful})});
  const d=await r.json();btn.parentElement.querySelector(".w").textContent=r.ok?"thanks · weight "+(d.weights&&d.weights[model]!=null?d.weights[model].toFixed(2):"–"):"vote failed";}catch(e){}
}
$("ask").onclick=async()=>{
  const question=$("q").value.trim();if(!question){$("status").textContent="Type a question first.";return}
  const passages=$("src").value.split("\\n").map(s=>s.trim()).filter(Boolean).map((t,i)=>({id:"s"+(i+1),text:t}));
  $("ask").disabled=true;$("status").textContent="Asking the crowd…";qid="q"+Date.now();
  const t0=performance.now();
  try{
    const r=await fetch("/",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question,passages})});
    const d=await r.json();
    if(!r.ok){$("status").textContent=d.error||("Error "+r.status);return}
    const rc=d.reconcile||{},conf=Math.round((rc.confidence||0)*100);
    $("bar").style.width=conf+"%";
    const n=(d.answers||[]).filter(a=>a.answer).length;
    $("conf").textContent=`${conf}% of the ${n} answering models share at least one claim`;
    $("agree").innerHTML=(rc.agreement||[]).map(c=>`<li>${esc(c)}${passages.length?srcTag(c,d.grounded):""}</li>`).join("")||"<li class='hint'>No claim was shared by two or more models.</li>";
    $("dissent").innerHTML=(rc.dissent||[]).slice(0,8).map(c=>`<li>${esc(c)}${passages.length?srcTag(c,d.grounded):""}</li>`).join("")||"<li class='hint'>None.</li>";
    $("cards").innerHTML=(d.answers||[]).map(a=>{const org=a.model.split("/")[0],name=a.model.split("/").pop();
      return `<div class="card"><h3>${esc(name)}</h3><div class="maker">${esc(MAKERS[org]||org)}</div>${a.answer?`<p>${esc(a.answer)}</p>`:`<p class="err">No answer: ${esc(a.error||"")}</p>`}
      <div class="votes"><button class="ghost" data-m="${esc(a.model)}" data-h="1">Helpful</button><button class="ghost" data-m="${esc(a.model)}" data-h="0">Not helpful</button><span class="w"></span></div></div>`}).join("");
    document.querySelectorAll("#cards button").forEach(b=>b.onclick=()=>vote(b.dataset.m,b.dataset.h==="1",b));
    $("out").hidden=false;$("status").textContent=`Answered in ${((performance.now()-t0)/1000).toFixed(1)} s`;
  }catch(e){$("status").textContent="Request failed: "+e.message}
  finally{$("ask").disabled=false}
};
</script>
</body>
</html>"""


def _build_response(
    status_code: int,
    body: str,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    headers = dict(CORS_HEADERS)
    if body.lstrip().startswith("<!DOCTYPE"):
        headers["Content-Type"] = "text/html; charset=utf-8"
    else:
        headers["Content-Type"] = "application/json"
    if extra_headers:
        headers.update(extra_headers)
    return {
        "statusCode": status_code,
        "headers": headers,
        "body": body,
    }


def _get_models() -> list[str]:
    raw = os.environ.get("KENNEL_MODELS", DEFAULT_MODELS)
    return [m.strip() for m in raw.split(",") if m.strip()]


def _get_api_key() -> str:
    return os.environ.get("KENNEL_API_KEY", "")


def _get_base_url() -> str:
    return os.environ.get("KENNEL_BASE_URL", DEFAULT_BASE_URL)


SYSTEM_PROMPT = ("Answer in 3 to 5 short, plain factual sentences. "
                 "No markdown, no lists, no headings, no greetings.")


def _ask_fn_default(model: str, question: str) -> str:
    from kennel.client import chat

    base = _get_base_url()
    key = _get_api_key()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return chat(base, key, model, messages, timeout=40, max_tokens=300, temperature=0.2)


def _validate_question(body: dict) -> Optional[str]:
    question = body.get("question")
    if not isinstance(question, str) or not question.strip():
        return "Missing or empty 'question' field."
    if len(question) > 4000:
        return "'question' exceeds 4000 characters."
    return None


def _validate_vote(body: dict, models: list[str]) -> Optional[str]:
    question_id = body.get("question_id")
    if not isinstance(question_id, str) or not question_id.strip():
        return "Missing or empty 'question_id' field."
    model = body.get("model")
    if not isinstance(model, str) or model not in models:
        return f"'model' must be one of {models}."
    helpful = body.get("helpful")
    if not isinstance(helpful, bool):
        return "'helpful' must be a boolean."
    return None


def lambda_handler(
    event: Dict[str, Any],
    context: Any,
    ask_fn: Any | None = None,
) -> Dict[str, Any]:
    """
    Handle a Lambda Function URL event.

    Supports both v1 (httpMethod/path) and v2 (requestContext.http) event formats.
    """
    # Detect HTTP method and path in either event shape
    http_method = (
        event.get("httpMethod")
        or event.get("requestContext", {})
        .get("http", {})
        .get("method", "")
    )
    raw_path = (
        event.get("path")
        or event.get("requestContext", {})
        .get("http", {})
        .get("path", "/")
    )
    path = raw_path.rstrip("/") or "/"

    if http_method == "OPTIONS":
        return _build_response(204, "")

    if http_method == "GET" and path == "/":
        return _build_response(200, HTML_PAGE)

    if http_method == "POST" and path == "/vote":
        models = _get_models()
        try:
            body_str = event.get("body", "{}")
            if isinstance(body_str, bytes):
                body_str = body_str.decode("utf-8")
            body = json.loads(body_str)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _build_response(400, json.dumps({"error": "Invalid JSON body."}))

        validation_error = _validate_vote(body, models)
        if validation_error:
            return _build_response(400, json.dumps({"error": validation_error}))

        votes_dir = os.environ.get("KENNEL_VOTES_DIR", "kennel_votes")
        from kennel.rsi import Votes

        store = Votes(votes_dir)
        store.record(body["question_id"], body["model"], body["helpful"])
        weights = store.weights(models)
        return _build_response(200, json.dumps({"weights": weights}))

    if http_method == "POST" and path == "/":
        api_key = _get_api_key()
        if not api_key and ask_fn is None:
            return _build_response(
                503,
                json.dumps({"error": "Service unavailable: no API key configured."}),
            )

        models = _get_models()
        if len(models) < 2:
            return _build_response(
                500,
                json.dumps({"error": "At least two models are required."}),
            )

        try:
            body_str = event.get("body", "{}")
            if isinstance(body_str, bytes):
                body_str = body_str.decode("utf-8")
            body = json.loads(body_str)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _build_response(400, json.dumps({"error": "Invalid JSON body."}))

        validation_error = _validate_question(body)
        if validation_error:
            return _build_response(400, json.dumps({"error": validation_error}))

        question = body["question"].strip()
        passages = body.get("passages", [])

        selected_ask_fn = ask_fn if ask_fn is not None else _ask_fn_default

        from kennel.crowd import ask_crowd

        crowd_answers = ask_crowd(question, models, selected_ask_fn)

        from kennel.reconcile import reconcile

        reconciliation = reconcile(crowd_answers)

        agreement_claims = reconciliation.get("agreement", [])
        dissent_claims = reconciliation.get("dissent", [])
        all_claims = agreement_claims + dissent_claims

        grounded_claims = []
        if passages and all_claims:
            from kennel.ground import ground

            grounded_claims = ground(all_claims, passages)

        vote_dir = os.environ.get("KENNEL_VOTES_DIR", "kennel_votes")
        from kennel.rsi import Votes

        store = Votes(vote_dir)
        weights = store.weights(models)

        response_body = {
            "answers": crowd_answers,
            "reconcile": reconciliation,
            "grounded": grounded_claims,
            "weights": weights,
        }

        return _build_response(200, json.dumps(response_body))

    return _build_response(404, json.dumps({"error": "Not found."}))
