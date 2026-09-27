from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json

VLLM = "http://127.0.0.1:8000"
HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Local chat</title><style>
:root{color-scheme:dark;--bg:#111416;--panel:#1b2023;--line:#364047;--text:#edf1f2;--muted:#9ca8ad;--accent:#e6b35a;--user:#29414b}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,#293b35,transparent 45%),var(--bg);color:var(--text);font:16px system-ui,sans-serif}main{max-width:900px;min-height:100vh;margin:auto;padding:24px 16px;display:flex;flex-direction:column}header{border-bottom:1px solid var(--line);padding-bottom:16px;margin-bottom:18px;display:flex;justify-content:space-between}h1{font-size:25px;margin:0}p{margin:4px 0;color:var(--muted);font-size:13px}.status{color:var(--accent);font-size:13px}.chat{flex:1;display:flex;flex-direction:column;gap:12px}.empty{margin:auto;color:var(--muted)}.msg{max-width:82%;padding:12px 15px;border:1px solid var(--line);border-radius:8px;white-space:pre-wrap}.user{align-self:flex-end;background:var(--user)}.bot{align-self:flex-start;background:var(--panel)}form{display:flex;gap:10px;border-top:1px solid var(--line);padding-top:15px}textarea{flex:1;min-height:54px;resize:vertical;background:var(--panel);border:1px solid var(--line);border-radius:7px;color:var(--text);padding:12px;font:inherit}button{background:var(--accent);border:0;border-radius:7px;padding:0 20px;font-weight:bold;cursor:pointer}button:disabled{opacity:.5}.foot{display:flex;justify-content:space-between;color:var(--muted);font-size:12px;margin-top:8px}@media(max-width:600px){.msg{max-width:95%}button{padding:0 14px}}
</style></head><body><main><header><div><h1>Local chat</h1><p>Qwen on vLLM + ROCm</p></div><div id="status" class="status">Connecting...</div></header><section id="chat" class="chat"><div class="empty">Ask the model a question.</div></section><form id="form"><textarea id="input" placeholder="Write your message..."></textarea><button id="send">Send</button></form><div class="foot"><span>vLLM: localhost:8000</span><span id="model"></span></div></main><script>
const chat=document.querySelector('#chat'),input=document.querySelector('#input'),send=document.querySelector('#send'),status=document.querySelector('#status'),modelEl=document.querySelector('#model');let model='Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4',messages=[];function add(text,kind){document.querySelector('.empty')?.remove();let x=document.createElement('div');x.className='msg '+kind;x.textContent=text;chat.append(x);chat.scrollTop=chat.scrollHeight;return x}async function boot(){try{let r=await fetch('/api/models');let d=await r.json();model=d.data?.[0]?.id||model;modelEl.textContent=model;status.textContent='Ready'}catch(e){status.textContent='vLLM unreachable'}}document.querySelector('#form').onsubmit=async e=>{e.preventDefault();let text=input.value.trim();if(!text||send.disabled)return;input.value='';add(text,'user');messages.push({role:'user',content:text});send.disabled=true;status.textContent='Generating...';let out=add('','bot');try{let r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model,messages,max_tokens:1024,temperature:.7})});let d=await r.json();if(!r.ok)throw Error(d.error?.message||JSON.stringify(d));let answer=d.choices?.[0]?.message?.content||'(empty response)';out.textContent=answer;messages.push({role:'assistant',content:answer})}catch(e){out.textContent='Error: '+e.message}send.disabled=false;status.textContent='Ready';input.focus()};input.onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();document.querySelector('#form').requestSubmit()}};boot();
</script></body></html>'''

def call_vllm(path, body=None):
    req = Request(VLLM + path, data=body, method="POST" if body else "GET")
    if body:
        req.add_header("Content-Type", "application/json")
    with urlopen(req, timeout=600) as response:
        return response.status, response.read(), response.headers.get("Content-Type", "application/json")

class Handler(BaseHTTPRequestHandler):
    def reply(self, status, data, content_type):
        self.send_response(status); self.send_header("Content-Type", content_type); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.reply(200, HTML.encode(), "text/html; charset=utf-8"); return
        if self.path == "/api/models":
            try: self.reply(*call_vllm("/v1/models"))
            except (HTTPError, URLError) as e: self.send_error(502, str(e))
            return
        self.send_error(404)
    def do_POST(self):
        if self.path != "/api/chat": self.send_error(404); return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try: self.reply(*call_vllm("/v1/chat/completions", body))
        except HTTPError as e: self.reply(e.code, e.read(), "application/json")
        except URLError as e: self.send_error(502, str(e))
    def log_message(self, fmt, *args): print("[chat-ui] " + fmt % args)

if __name__ == "__main__":
    print("Chat UI: http://localhost:8080")
    ThreadingHTTPServer(("127.0.0.1", 8080), Handler).serve_forever()
