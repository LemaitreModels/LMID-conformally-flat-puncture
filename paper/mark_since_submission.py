"""Mark in red, inside paper.tex itself, every piece of text that is not in the submitted paper.

    python mark_since_submission.py            # re-mark paper.tex against the submission
    python mark_since_submission.py --strip    # remove ALL markup (red and purple): the clean source
    python mark_since_submission.py --words    # mark only the changed words, not whole sentences

Run it from ``paper/``.  The submitted text is read from git, commit ``SUBMITTED``
(b0f2f39, the PRD submission of 2026-08-24); ``--submitted FILE`` and ``--paper FILE``
override the two inputs.

* Idempotent: existing markup is stripped first, so re-running after edits re-marks against
  the submission (text you add later is then red as well).
* The source is otherwise untouched: stripping the markup gives back the input BYTE FOR BYTE,
  and the script refuses to write unless that holds.
* Granularity: a sentence of running text with any change is red as a whole (``--words``
  marks only the changed words).  Display math is coloured as a whole block.
* Markup: ``\\newtext{...}`` around a changed span that is safe to wrap; otherwise the whole
  line is bracketed by the switches ``\\newtextbegin`` ... ``\\newtextend``.  A
  ``\\shownewfalse`` in the preamble turns all of it black.
* Referee changes are PURPLE and written by hand: ``\\refereetext{...}`` inline, or
  ``\\refereebegin`` ... ``\\refereeend{}`` around a block or a display.  They are part of the
  source, so a re-mark keeps them and never colours red inside them (purple wins);
  ``\\showrefereefalse`` turns them black, and ``--strip`` removes them with the red markup.
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
    "% purple = changed in reply to the referee (hand-written \\refereetext / \\refereebegin..\\refereeend)\n"
    "\\newif\\ifshowreferee \\showrefereetrue\n"
    "\\definecolor{refereepurple}{rgb}{0.45,0.1,0.75}\n"
    "\\newcommand{\\refereetext}[1]{\\ifshowreferee{\\color{refereepurple}#1}\\else#1\\fi}\n"
    "\\newcommand{\\refereebegin}{\\ifshowreferee\\color{refereepurple}\\fi}\n"
    "\\newcommand{\\refereeend}{\\ifshowreferee\\color{black}\\fi}\n"
    "% --- end new-since-submission markup ---\n"
)
OPEN, B, E = "\\newtext{", "\\newtextbegin ", "\\newtextend "
EL = "\\newtextend{}"   # at a line end: the {} keeps the newline a space
NOWRAP = {"\\label", "\\ref", "\\eqref", "\\cite", "\\citep", "\\citet", "\\includegraphics",
          "\\begin", "\\end", "\\input", "\\url", "\\href", "\\bibliography", "\\bibliographystyle",
          "\\documentclass", "\\usepackage", "\\newcommand", "\\renewcommand", "\\label"}
R_OPEN, R_B, R_E = "\\refereetext{", "\\refereebegin", "\\refereeend"
TOK = re.compile(r"\s+|\\[A-Za-z@]+\*?|\\.|[A-Za-z0-9]+|.", re.S)


def _close(t, k):
    """Index just past the brace that closes the group opened right before ``k``."""
    d = 1
    while d:
        c = t[k]
        if c == "\\":
            k += 2
            continue
        d += (c == "{") - (c == "}")
        k += 1
    return k


def referee_mask(t):
    """True on every character of a hand-written referee (purple) span, macros included."""
    mask = [False] * len(t)
    for m in re.finditer(re.escape(R_OPEN), t):
        for c in range(m.start(), _close(t, m.end())):
            mask[c] = True
    for m in re.finditer(re.escape(R_B) + r"\b", t):
        e = t.find(R_E, m.end())
        assert e >= 0, f"\\refereebegin at offset {m.start()} has no \\refereeend"
        end = e + len(R_E) + (2 if t.startswith("{}", e + len(R_E)) else 0)
        for c in range(m.start(), end):
            mask[c] = True
    return mask


def strip_referee(t):
    t = re.sub(re.escape(R_B) + r"\s?", "", t)
    t = t.replace(R_E + "{}", "").replace(R_E + " ", "").replace(R_E, "")
    return _unwrap(t, R_OPEN)


def _unwrap(t, opener):
    out, i = [], 0
    while True:
        j = t.find(opener, i)
        if j < 0:
            out.append(t[i:])
            break
        out.append(t[i:j])
        k = _close(t, j + len(opener))
        out.append(t[j + len(opener):k - 1])
        i = k
    return "".join(out)


DEFS_RX = re.compile(r"% --- new-since-submission markup.*?% --- end new-since-submission markup ---\n",
                     re.S)


def strip(t):
    t = DEFS_RX.sub("", t)
    t = t.replace(B, "").replace(EL, "").replace(E, "").replace(E.rstrip(), "")
    return _unwrap(t, OPEN)


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
    # a sectioning command inside a brace group would number in a local scope, so the
    # \label after it reads a stale counter: colour those lines with the switches instead
    if any(cmd in seg for cmd in ("\\begin", "\\end", "\\label", "\\item", "\\caption",
                                  "\\section", "\\subsection", "\\subsubsection")):
        return False
    stack = _depth_and_cmd(line, a)
    if any(c in NOWRAP for c in stack if c):
        return False
    if in_display and stack:            # inside a brace group in display math: leave it
        return False
    return True


STRUCT = re.compile(r"\s*(%|\\(begin|end|section|subsection|subsubsection|paragraph|label|"
                    r"includegraphics|centering|maketitle|appendix|bibliography|item)\b|\\\[|\\\])")
ABBREV = re.compile(r"(\b(al|Eq|Eqs|Ref|Refs|Fig|Figs|Sec|Secs|App|Tab|cf|vs|resp)|e\.g|i\.e)\.$")
SENT_END = re.compile(r"[.?!](?=\s)")


def _whole_sentences(cur, flag, sub=None, purple=None):
    """Widen every change in running text to the whole sentence that contains it.

    Running text is split into blocks at blank lines, display math and structural lines
    (``\\begin``, ``\\section``, ``\\label``, comments, ...), which are never widened into;
    within a block a sentence ends at ``.``/``?``/``!`` followed by whitespace, except after
    the abbreviations in ``ABBREV``.  A ``\\caption{`` opens a sentence.
    """
    flag = flag[:]
    lines = cur.splitlines(True)
    text, pos, in_doc, display = [], 0, False, 0
    for line in lines:
        body = line.rstrip("\n")
        if "\\begin{document}" in body:
            in_doc = True
        opens = len(re.findall(r"\\begin\{(equation|align|gather|multline|eqnarray)\*?\}|\\\[", body))
        closes = len(re.findall(r"\\end\{(equation|align|gather|multline|eqnarray)\*?\}|\\\]", body))
        is_text = (in_doc and body.strip() and display == 0 and opens == 0 and closes == 0
                   and not STRUCT.match(body))
        text.append((pos, pos + len(line), is_text))
        display += opens - closes
        pos += len(line)
    blocks, k = [], 0
    while k < len(text):
        if not text[k][2]:
            k += 1
            continue
        a = text[k][0]
        while k < len(text) and text[k][2]:
            k += 1
        blocks.append((a, text[k - 1][1]))
    for a, b in blocks:
        cuts = [a]
        for m in re.finditer(r"\\caption\{", cur[a:b]):
            cuts.append(a + m.end())
        for m in SENT_END.finditer(cur, a, b):
            if not ABBREV.search(cur[max(a, m.start() - 8):m.end()]):
                cuts.append(m.end())
        cuts = sorted(set(cuts)) + [b]
        for s, e in zip(cuts, cuts[1:]):
            if any(flag[s:e]):
                if sub is not None and _unchanged_outside(cur, s, e, sub, purple):
                    for c in range(s, e):
                        flag[c] = False
                    continue
                for c in range(s, e):
                    if not cur[c].isspace():
                        flag[c] = True
    return flag


def _unchanged_outside(cur, s, e, sub, purple):
    """Whether the sentence cur[s:e] occurs in the submitted text, purple spans aside.

    The token diff can pair a common word at a line start ("The") with the same word of a
    neighbouring new line, and so flag a sentence nobody changed; and a sentence changed only
    inside a referee span is not a red change.  Both are settled by looking the sentence up:
    its text outside purple spans, whitespace-insensitive, with each purple span as a wildcard.

    The lookup finds the non-purple pieces in order with ``str.find``, each starting where the
    previous one ended: the leftmost occurrence of each piece leaves the most room for the
    rest, so this answers the same question as one regex with a lazy wildcard per purple span,
    in linear time.  That regex backtracked without end when a "sentence" held several purple
    spans and did not occur in the submission (two purple sentences end in ``.}``, which is no
    sentence end, so they and the next red sentence form one span).
    """
    pieces, k = [], s
    while k < e:
        p = purple[k] if purple else False
        j = k
        while j < e and (purple[j] if purple else False) == p:
            j += 1
        if not p:
            piece = " ".join(cur[k:j].split())
            if piece:
                pieces.append(piece)
        k = j
    if not pieces:
        return False
    hay = _normalized(sub)
    at = 0
    for piece in pieces:
        hit = hay.find(piece, at)
        if hit < 0:
            return False
        at = hit + len(piece)
    return True


def _moved_displays(cur, flag, sub):
    """Unflag every display environment that occurs verbatim (whitespace aside) in the submission.

    The line diff is sequential, so a display moved unchanged to another place (e.g. into an
    appendix) reads as deleted and re-inserted and would be coloured.  Moved running text is
    settled by the sentence lookup in ``_whole_sentences``; this is the same lookup for a
    display, taken as one block from its ``\\begin`` line to its ``\\end`` line.
    """
    flag = flag[:]
    pos, display, start = 0, 0, None
    hay = _normalized(sub)
    for line in cur.splitlines(True):
        body = line.rstrip("\n")
        opens = len(re.findall(r"\\begin\{(equation|align|gather|multline|eqnarray)\*?\}|\\\[", body))
        closes = len(re.findall(r"\\end\{(equation|align|gather|multline|eqnarray)\*?\}|\\\]", body))
        if display == 0 and opens > 0:
            start = pos
        display += opens - closes
        pos += len(line)
        if start is not None and display == 0:
            s, e = start, pos
            start = None
            if any(flag[s:e]) and " ".join(cur[s:e].split()) in hay:
                for c in range(s, e):
                    flag[c] = False
    return flag


_NORMALIZED = {}


def _normalized(t):
    """``t`` with every whitespace run collapsed to one space, computed once per text."""
    key = id(t)
    if _NORMALIZED.get(key, (None,))[0] is not t:
        _NORMALIZED[key] = (t, " ".join(t.split()))
    return _NORMALIZED[key][1]


def mark(sub, cur):
    # Anchor on whole lines first, then diff tokens only inside each changed block: a single
    # token-level diff over the whole file can pair a common word ("The") with a far-away
    # occurrence and flag it in an otherwise unchanged sentence.
    old_l, new_l = sub.splitlines(True), cur.splitlines(True)
    line_start, p = [], 0
    for ln in new_l:
        line_start.append(p)
        p += len(ln)
    flag = [False] * len(cur)
    lsm = difflib.SequenceMatcher(None, old_l, new_l, autojunk=False)
    for ltag, a1, a2, b1, b2 in lsm.get_opcodes():
        if ltag == "equal" or b1 == b2:
            continue
        old, new = TOK.findall("".join(old_l[a1:a2])), TOK.findall("".join(new_l[b1:b2]))
        starts, q = [], line_start[b1]
        for tk in new:
            starts.append(q)
            q += len(tk)
        sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ("insert", "replace"):
                for j in range(j1, j2):
                    if new[j].strip():
                        for c in range(starts[j], starts[j] + len(new[j])):
                            flag[c] = True
    purple = referee_mask(cur)
    flag = [f and not p for f, p in zip(flag, purple)]
    if WHOLE_SENTENCES:
        flag = _whole_sentences(cur, flag, sub, purple)
        flag = [f and not p for f, p in zip(flag, purple)]
    flag = _moved_displays(cur, flag, sub)
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
        # a float's placement option ([tp]) is not text: \newtext inside it breaks the float
        float_open = re.match(r"\s*\\begin\{(figure|table)\*?\}", body)
        if not in_doc or not any(fl) or body.lstrip().startswith("%") or float_open:
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
        # no {} here: after a display the newline must vanish, and an unexpandable {} would
        # stop LaTeX's \ignorespacesafterend and indent the next line by a space
        out[last] = lb + E.rstrip() + out[last][len(lb):]
        stats["display_blocks"] += 1
    res = "".join(out)
    i = res.index("\\begin{document}")
    res = res[:i] + DEFS + res[i:]
    return res, stats


SUBMITTED = "b0f2f39"
WHOLE_SENTENCES = "--words" not in sys.argv


def _arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


if __name__ == "__main__":
    import os
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    cur_path = _arg("--paper", os.path.join(here, "paper.tex"))
    cur = strip(open(cur_path).read())
    if "--strip" in sys.argv:
        cur = strip_referee(cur)
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
