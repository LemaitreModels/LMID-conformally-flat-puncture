"""Mark in red, inside paper.tex itself, every piece of text that is not in the submitted paper.

    python mark_since_submission.py            # re-mark paper.tex against the submission
    python mark_since_submission.py --strip    # remove the markup, leaving the clean source

Run it from ``paper/``.  The submitted text is read from git, commit ``SUBMITTED``
(b0f2f39, the PRD submission of 2026-08-24); ``--submitted FILE`` and ``--paper FILE``
override the two inputs.

* Idempotent: existing markup is stripped first, so re-running after edits re-marks against
  the submission (text you add later is then red as well).
* The source is otherwise untouched: stripping the markup gives back the input BYTE FOR BYTE,
  and the script refuses to write unless that holds.
* Markup: ``\\newtext{...}`` around a changed span that is safe to wrap; otherwise the whole
  line is bracketed by the switches ``\\newtextbegin`` ... ``\\newtextend``.  A
  ``\\shownewfalse`` in the preamble turns all of it black.
"""
import difflib
import re
import sys

DEFS = (
    "% --- new-since-submission markup (mark_since_submission.py; red = not in the submitted paper) ---\n"
    "\\newif\\ifshownew \\shownewtrue\n"
    "\\newcommand{\\newtext}[1]{\\ifshownew{\\color{red}#1}\\else#1\\fi}\n"
    "\\newcommand{\\newtextbegin}{\\ifshownew\\color{red}\\fi}\n"
    "\\newcommand{\\newtextend}{\\ifshownew\\color{black}\\fi}\n"
    "% --- end new-since-submission markup ---\n"
)
OPEN, B, E = "\\newtext{", "\\newtextbegin ", "\\newtextend "
EL = "\\newtextend{}"   # at a line end: the {} keeps the newline a space
NOWRAP = {"\\label", "\\ref", "\\eqref", "\\cite", "\\citep", "\\citet", "\\includegraphics",
          "\\begin", "\\end", "\\input", "\\url", "\\href", "\\bibliography", "\\bibliographystyle",
          "\\documentclass", "\\usepackage", "\\newcommand", "\\renewcommand", "\\label"}
TOK = re.compile(r"\s+|\\[A-Za-z@]+\*?|\\.|[A-Za-z0-9]+|.", re.S)


def strip(t):
    t = t.replace(DEFS, "")
    t = t.replace(B, "").replace(EL, "").replace(E, "").replace(E.rstrip(), "")
    out, i = [], 0
    while True:
        j = t.find(OPEN, i)
        if j < 0:
            out.append(t[i:])
            break
        out.append(t[i:j])
        k, d = j + len(OPEN), 1
        start = k
        while d:
            c = t[k]
            if c == "\\":
                k += 2
                continue
            d += (c == "{") - (c == "}")
            k += 1
        out.append(t[start:k - 1])
        i = k
    return "".join(out)


def _depth_and_cmd(line, pos):
    """Brace depth at ``pos`` within the line, and the command owning the innermost open brace."""
    stack, k = [], 0
    while k < pos:
        c = line[k]
        if c == "\\":
            m = re.match(r"\\[A-Za-z@]+\*?", line[k:])
            k += len(m.group(0)) if m else 2
            continue
        if c == "{":
            m = re.search(r"(\\[A-Za-z@]+\*?)\s*(\[[^\]]*\])?\s*$", line[:k])
            stack.append(m.group(1) if m else None)
        elif c == "}" and stack:
            stack.pop()
        k += 1
    return stack


def _dollars(s):
    return len(re.findall(r"(?<!\\)\$", s))


def _balanced(s):
    d = 0
    k = 0
    while k < len(s):
        c = s[k]
        if c == "\\":
            k += 2
            continue
        d += (c == "{") - (c == "}")
        if d < 0:
            return False
        k += 1
    return d == 0


def _safe(line, a, b, in_display):
    seg = line[a:b]
    if not seg.strip() or "\n" in seg or "%" in seg.replace("\\%", ""):
        return False
    if not _balanced(seg) or _dollars(seg) % 2:
        return False
    if in_display or _dollars(line[:a]) % 2:  # starts inside math: wrapping can split it
        return False
    if seg.rstrip()[-1:] in "^_":
        return False
    if "&" in seg.replace("\\&", "") or "\\\\" in seg:
        return False
    if any(cmd in seg for cmd in ("\\begin", "\\end", "\\label", "\\item", "\\caption")):
        return False
    stack = _depth_and_cmd(line, a)
    if any(c in NOWRAP for c in stack if c):
        return False
    if in_display and stack:            # inside a brace group in display math: leave it
        return False
    return True


def mark(sub, cur):
    old, new = TOK.findall(sub), TOK.findall(cur)
    assert "".join(new) == cur
    starts, p = [], 0
    for t in new:
        starts.append(p)
        p += len(t)
    flag = [False] * len(cur)
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            for j in range(j1, j2):
                if new[j].strip():
                    for c in range(starts[j], starts[j] + len(new[j])):
                        flag[c] = True
    lines = cur.splitlines(True)
    out, pos, in_doc, display = [], 0, False, 0
    env_start, env_line, env_hit, env_id = [], [], set(), -1
    stats = dict(spans=0, lines=0, display_blocks=0)
    for line in lines:
        n = len(line)
        fl = flag[pos:pos + n]
        body = line.rstrip("\n")
        if "\\begin{document}" in body:
            in_doc = True
        opens = len(re.findall(r"\\begin\{(equation|align|gather|multline|eqnarray)\*?\}|\\\[", body))
        closes = len(re.findall(r"\\end\{(equation|align|gather|multline|eqnarray)\*?\}|\\\]", body))
        in_display = display > 0 or opens > 0
        if display == 0 and opens > 0:
            env_id = len(env_start)
            env_start.append(len(out))
        if in_display:
            env_line.append((env_id, len(out)))
        if not in_doc or not any(fl) or body.lstrip().startswith("%"):
            out.append(line)
        else:
            # flagged runs, merging gaps of pure whitespace
            runs, k = [], 0
            while k < len(body):
                if fl[k]:
                    a = k
                    while k < len(body) and (fl[k] or (body[k].isspace() and any(fl[k:k + 40]))):
                        k += 1
                    b = k
                    while b > a and body[b - 1].isspace():
                        b -= 1
                    runs.append((a, b))
                else:
                    k += 1
            if runs and all(_safe(body, a, b, in_display) for a, b in runs):
                piece, last = [], 0
                for a, b in runs:
                    piece += [body[last:a], OPEN, body[a:b], "}"]
                    last = b
                piece.append(body[last:])
                out.append("".join(piece) + line[len(body):])
                stats["spans"] += len(runs)
            elif in_display:
                out.append(line)                    # display math: coloured as a block below
                env_hit.add(env_id)
            else:
                lead = len(body) - len(body.lstrip())
                out.append(body[:lead] + B + body[lead:] + EL + line[len(body):])
                stats["lines"] += 1
        display += opens - closes
        pos += n
    # a changed display environment is coloured whole: switch on before its first line,
    # off after its last
    for eid in sorted(env_hit):
        idx = [li for e, li in env_line if e == eid]
        first, last = min(idx), max(idx)
        ln = out[first]
        lead = len(ln) - len(ln.lstrip())
        out[first] = ln[:lead] + B + ln[lead:]
        lb = out[last].rstrip("\n")
        out[last] = lb + EL + out[last][len(lb):]
        stats["display_blocks"] += 1
    res = "".join(out)
    i = res.index("\\begin{document}")
    res = res[:i] + DEFS + res[i:]
    return res, stats


SUBMITTED = "b0f2f39"


def _arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


if __name__ == "__main__":
    import os
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    cur_path = _arg("--paper", os.path.join(here, "paper.tex"))
    cur = strip(open(cur_path).read())
    if "--strip" in sys.argv:
        open(cur_path, "w").write(cur)
        print("markup removed")
        sys.exit(0)
    sub_file = _arg("--submitted", None)
    sub = (open(sub_file).read() if sub_file else
           subprocess.run(["git", "show", f"{SUBMITTED}:paper/paper.tex"], cwd=here,
                          check=True, capture_output=True, text=True).stdout)
    res, stats = mark(sub, cur)
    assert strip(res) == cur, "stripping the markup does not give back the source; nothing written"
    open(cur_path, "w").write(res)
    print(f"marked: {stats['spans']} spans, {stats['lines']} whole lines, "
          f"{stats['display_blocks']} display-math blocks")
