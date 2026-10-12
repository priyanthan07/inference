import asyncio, os, time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
MODE = os.environ.get("MODE", "single")                 # "single" or "batch"
MAX_BATCH = int(os.environ.get("MAX_BATCH", "8"))       # batch mode: max requests per generate()
MAX_WAIT_MS = float(os.environ.get("MAX_WAIT_MS", "30"))  # batch mode: how long to wait to fill a batch
DTYPE = getattr(torch, os.environ.get("DTYPE", "float32"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GREEDY = dict(do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)
state = {}

class GenRequest(BaseModel):
    prompt: str               # plain user question; the server applies the chat template
    max_new_tokens: int = 48
    
def first_eos_cut(row, eos_ids):
    for i, t in enumerate(row):
        if t in eos_ids:
            return i + 1
    return len(row)

@torch.inference_mode()
def generate_batch(questions, max_news):
    """Runs in a worker thread. One padded generate() call for all questions."""
    tok, model = state["tok"], state["model"]
    texts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                     add_generation_prompt=True) for q in questions]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
    out = model.generate(**enc, max_new_tokens=max(max_news), pad_token_id=tok.pad_token_id, **GREEDY)
    
    new = out[:, enc.input_ids.shape[1]:].tolist()
    results = []
    for row, limit in zip(new, max_news):
        row = row[:limit]
        n = first_eos_cut(row, state["eos_ids"])
        results.append((tok.decode(row[:n], skip_special_tokens=True), n))
        
    return results

async def batch_worker():
    """Batch mode: collect requests for up to MAX_WAIT_MS (or MAX_BATCH), then run them together."""
    queue = state["queue"]
    
    while True:
        first = await queue.get()                          # wait for at least one request
        batch = [first]
        deadline = time.perf_counter() + MAX_WAIT_MS / 1000
        
        while len(batch) < MAX_BATCH:
            try:                                           # 1) take everything already waiting, no delay
                batch.append(queue.get_nowait())
                continue
            except asyncio.QueueEmpty:
                pass
            
            remaining = deadline - time.perf_counter()     # 2) then wait up to MAX_WAIT_MS for more
            
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(queue.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break
            
        state["batch_sizes"].append(len(batch))
        started = time.perf_counter()
        
        try:
            results = await asyncio.to_thread(generate_batch, [b["q"] for b in batch], [b["max_new"] for b in batch])
            gen_s = time.perf_counter() - started
            for b, (text, n) in zip(batch, results):
                b["future"].set_result({"text": text, "new_tokens": n, "gen_s": gen_s,
                                        "queue_wait_s": started - b["arrived"], "batch_size": len(batch)})
        except Exception as e:                             # never leave callers waiting forever
            for b in batch:
                if not b["future"].done():
                    b["future"].set_exception(e)
                    
@asynccontextmanager
async def lifespan(app):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"                              # set ONCE at startup
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE).eval()
    eos = model.generation_config.eos_token_id
    state.update(tok=tok, model=model, eos_ids=set(eos if isinstance(eos, list) else [eos]),
                 lock=asyncio.Lock(), queue=asyncio.Queue(), batch_sizes=[])
    worker = asyncio.create_task(batch_worker()) if MODE == "batch" else None
    yield
    if worker:
        worker.cancel()
        
app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "mode": MODE, "max_batch": MAX_BATCH, "max_wait_ms": MAX_WAIT_MS}


@app.get("/stats")
async def stats():
    sizes = state["batch_sizes"]
    return {"batches": len(sizes), "batch_sizes": sizes}

@app.post("/generate")
async def generate(req: GenRequest):
    arrived = time.perf_counter()
    if MODE == "single":
        # one generate() at a time: requests wait for the lock
        async with state["lock"]:
            started = time.perf_counter()
            state["batch_sizes"].append(1)
            (text, n), = await asyncio.to_thread(generate_batch, [req.prompt], [req.max_new_tokens])
            return {"text": text, "new_tokens": n, "gen_s": time.perf_counter() - started,
                    "queue_wait_s": started - arrived, "batch_size": 1}
    future = asyncio.get_running_loop().create_future()
    await state["queue"].put({"q": req.prompt, "max_new": req.max_new_tokens,
                              "arrived": arrived, "future": future})
    return await future
