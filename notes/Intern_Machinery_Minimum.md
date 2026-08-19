# Bare Minimum Machinery — ML Research Intern

*The goal is not mastery. The goal is that the machinery stops being the bottleneck,
so time goes to the parts that actually matter: reading, hypotheses, design, results.*

*Scope check: nobody expects an intern to know all of PyTorch, or to write production
code. They expect you to run experiments without supervision, not lose data, and be
able to say what your code does. That's it.*

---

## The honest bar

An intern is useful when they can be handed **"run this experiment, here are the
configs, report back"** and come back three days later with results, logs, and a
sentence about what broke. Everything below serves that.

You are NOT expected to: write CUDA, design distributed training, know
architecture internals, write production-quality software, or have memorised APIs.

---

## Tier 0 — Non-negotiable

### Python (the working subset)
- Lists, dicts, sets, tuples. When to use which.
- Comprehensions: `[f(x) for x in xs if cond]`
- `defaultdict`, `Counter` — you will use these constantly for analysis
- File I/O, and specifically **JSONL** (one JSON object per line) — the default
  format for experiment output
- `argparse` — every script takes config from the command line, not hardcoded
- Exceptions: `try/except`, and knowing when NOT to catch (silent failures kill experiments)
- f-strings, string formatting
- Regex basics: `re.search`, `re.findall`, capture groups, `(?m)` multiline

✅ *You've already done all of this in Checkpoint 1.*

### Bash
- Navigate: `cd`, `ls`, `pwd`, `mkdir -p`, `cp`, `mv`, `rm`
- Inspect files without opening them: `head`, `tail`, `wc -l`, `cat`, `less`
- Pipes and `grep` — `grep "error" log.txt | wc -l`
- `nohup cmd &` or `tmux` — run something that survives your ssh session dropping
- `scp` / `rsync` — move files between your machine and a server
- `nvidia-smi` — is the GPU actually being used, how much memory

### Git
Only these, honestly:
- `clone`, `status`, `diff`, `add`, `commit -m`, `push`, `pull`
- `branch`, `checkout -b`, and how to open a pull request
- `.gitignore` — never commit data, checkpoints, or `.env`
- What a merge conflict looks like and how to resolve one by hand

That's 90% of daily git. Rebasing and cherry-picking can wait.

### Debugging
The actual skill, in order:
1. **Read the traceback bottom-up.** The last line is the error; the lines above
   are the call path. Most people panic and read top-down.
2. **Isolate.** Reproduce the failure in the smallest possible script.
3. **Print the shape/type/value** of the thing you assumed. Most bugs are a wrong
   assumption about data, not wrong logic.
4. `import pdb; pdb.set_trace()` or `breakpoint()` when prints aren't enough.
5. **Distrust silent success.** A script that "works" but returns defaults on
   failure (e.g. score 0.5 on parse failure) is worse than one that crashes.

---

## Tier 1 — The ML stack

### NumPy
- Array creation, `shape`, `reshape`, indexing, slicing, boolean masks
- Axis semantics: `arr.mean(axis=0)` vs `axis=1` — know which is which
- Broadcasting rules (why `(3,1) + (1,4)` gives `(3,4)`)
- `np.concatenate`, `np.stack`, `argmax`, `argsort`

### PyTorch (the intern subset)
- Tensors: creation, `.shape`, `.dtype`, `.device`
- Moving between CPU/GPU: `.to(device)`, and why mismatched devices error
- `torch.no_grad()` — and *why* it matters (no gradient graph = less memory)
- `model.eval()` vs `model.train()` — dropout/batchnorm behave differently
- dtypes: `float32`, `bfloat16`, `float16` — memory vs precision tradeoff
- Reading a training loop: forward → loss → `backward()` → `optimizer.step()` →
  `zero_grad()`. You should be able to explain each line.
- Common errors: CUDA OOM, device mismatch, shape mismatch. Know what each means.

### HuggingFace
- `AutoTokenizer`, `AutoModelForCausalLM`, `from_pretrained`
- `apply_chat_template` — instruct models need their own format
- `model.generate()` args: `max_new_tokens`, `do_sample`, `temperature`, `top_p`
- **Slicing off the prompt**: `output[0][input_len:]` — generate returns
  prompt + completion
- **Batched generation needs `padding_side="left"`** for decoder-only models.
  Right padding silently produces garbage with no error.
- `datasets.load_dataset`
- Quantization basics: `BitsAndBytesConfig`, 4-bit/8-bit, what it costs

✅ *All of the HuggingFace list you've now used directly.*

### Plotting
- `matplotlib`: line, scatter, histogram, `plt.savefig`
- Enough to make a figure for a paper. Not more.

---

## Tier 2 — Experiment hygiene

This is what separates an intern who is trusted from one who isn't.

- **Seeds.** Set `random`, `numpy`, `torch` seeds from one config value, AFTER
  arg parsing. Know which sources of randomness you're controlling.
- **Never hardcode config.** Every knob is a CLI arg or a config file.
- **Save raw, aggregate later.** Per-item outputs, not just summaries.
  Regenerating costs GPU; re-aggregating costs nothing.
- **Incremental writes + resume.** Append results as they're produced; skip
  already-done work on restart. Sessions die.
- **Record provenance.** Every output file says which model, which config,
  which commit produced it.
- **Count failures explicitly.** Never let a failure become a plausible default
  value in your data.
- **Pin versions.** `pip freeze > requirements.txt`. Library drift will break
  your code in three months.

✅ *Checkpoint 1 has all seven of these built in.*

### Environments
- `venv` or `conda` — one env per project
- `pip install -r requirements.txt`
- Knowing that CUDA version / torch version / driver must be compatible, and
  that this is the #1 cause of "it works on my machine"

---

## What you can skip (for now)

- Distributed training internals (DDP, FSDP, DeepSpeed) — learn when you need it
- CUDA kernels, Triton
- Production serving, Docker, CI/CD
- Advanced OOP / design patterns
- Most of pandas (know `read_csv`, `groupby`, `to_csv` and move on)
- Writing your own autograd

---

## Self-test — can you do these without looking them up?

1. Write a script that reads a JSONL file, filters rows, and writes a new JSONL.
2. Take a CLI arg for a seed and set all three RNGs from it.
3. Load an instruct model, generate from a batch of 8 prompts, and correctly
   strip the prompt from each output.
4. Explain why `padding_side="left"` matters, mechanically.
5. Given a traceback, name the file and line where the error occurred.
6. Make a script resume from partial output after a crash.
7. Branch, commit, push, open a PR.
8. Given a CUDA OOM, name three things you could change.
9. Explain what `torch.no_grad()` does and why it saves memory.
10. Read someone else's training loop and say what each line does.

**If you can do 1–7, you are not going to be embarrassed in a lab.**
8–10 are the difference between running experiments and designing them.

---

## How to train this

**Not with drills.** Toy problems come pre-formalised — the hard part is already
done for you. Build the machinery inside real work.

The reps come from: writing the script yourself first, hitting a bug, reading the
traceback, fixing it. That loop is the training. Every experiment you run generates
several of these for free.

One deliberate exercise worth doing *once*, later: reimplement from scratch, no
libraries beyond numpy — linear regression with hand-written gradient descent,
then a single attention head. Not for the result. For the mechanical fluency of
translating math into code.

---

## Where you actually stand

Covered in this project: JSONL pipelines, argparse, regex parsing, HF generate,
chat templates, batched inference, `padding_side`, quantized loading, incremental
writes, resume-on-crash, parse-failure tracking, provenance records.

Thinner: git workflow, bash/server work, PyTorch internals (you've used
inference, not training loops), plotting, environment management.

The gap is smaller than it feels. Most of what's left is literacy — it comes
fast with use, and nobody thinks about it once it's there.
