"""Сборка страницы квизов из quizzes/*.json в docs/index.html.

Вопросы встраиваются прямо в HTML: страница должна работать без сети и без
сервера (GitHub Pages, локальный файл, опубликованный артефакт).

    uv run python tools/build_quiz_page.py
    uv run python tools/build_quiz_page.py --out docs/index.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUIZ_DIR = ROOT / "quizzes"
DEFAULT_OUT = ROOT / "docs" / "index.html"

REPO_URL = "https://github.com/IlyaChichkanov/Reinforcement-learning"

TEMPLATE = """<title>Квизы курса RL</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,500;7..72,600&family=Golos+Text:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap">
<style>
  :root {
    --paper: #eef2f6;
    --surface: #ffffff;
    --surface-sunk: #f4f7fa;
    --ink: #10202d;
    --ink-soft: #4d6273;
    --line: #d3dee7;
    --line-strong: #b6c7d4;
    --ice: #22688f;
    --ice-soft: #dbeaf4;
    --amber: #9a6113;
    --amber-soft: #f6ecda;
    --good: #1f7a4d;
    --good-soft: #dff0e6;
    --bad: #a83c2a;
    --bad-soft: #f7e3de;
    --shadow: 0 1px 2px rgba(16, 32, 45, .06), 0 8px 24px -16px rgba(16, 32, 45, .35);
    color-scheme: light;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --paper: #0c151d;
      --surface: #131f29;
      --surface-sunk: #0f1922;
      --ink: #e4edf4;
      --ink-soft: #9db0c0;
      --line: #24343f;
      --line-strong: #354856;
      --ice: #79b8dd;
      --ice-soft: #17303f;
      --amber: #dda65a;
      --amber-soft: #2e2416;
      --good: #6ecb98;
      --good-soft: #14291f;
      --bad: #e58a74;
      --bad-soft: #2d1a16;
      --shadow: 0 1px 2px rgba(0, 0, 0, .4), 0 8px 24px -16px rgba(0, 0, 0, .8);
      color-scheme: dark;
    }
  }
  :root[data-theme="dark"] {
    --paper: #0c151d;
    --surface: #131f29;
    --surface-sunk: #0f1922;
    --ink: #e4edf4;
    --ink-soft: #9db0c0;
    --line: #24343f;
    --line-strong: #354856;
    --ice: #79b8dd;
    --ice-soft: #17303f;
    --amber: #dda65a;
    --amber-soft: #2e2416;
    --good: #6ecb98;
    --good-soft: #14291f;
    --bad: #e58a74;
    --bad-soft: #2d1a16;
    --shadow: 0 1px 2px rgba(0, 0, 0, .4), 0 8px 24px -16px rgba(0, 0, 0, .8);
    color-scheme: dark;
  }

  body {
    background: var(--paper);
    color: var(--ink);
    font-family: "Golos Text", ui-sans-serif, system-ui, "Segoe UI", sans-serif;
    font-size: 16px;
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 44rem; margin: 0 auto; padding-inline: 20px; padding-block: 0 72px; }
  h1, h2, h3 { font-family: Literata, Georgia, "Times New Roman", serif; font-weight: 600; text-wrap: balance; margin: 0; }
  p { margin: 0; }
  button { font: inherit; color: inherit; cursor: pointer; }
  a { color: var(--ice); }

  /* ---------- шапка ---------- */
  header {
    position: sticky; top: env(safe-area-inset-top, 0px); z-index: 20;
    background: color-mix(in srgb, var(--paper) 88%, transparent);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--line);
    margin-inline: -20px; padding: 14px 20px 0;
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .brand h1 { font-size: 1.15rem; letter-spacing: -.01em; }
  .brand .sub { color: var(--ink-soft); font-size: .8rem; line-height: 1.3; }
  .lake { flex: none; width: 40px; height: 40px; }
  .switches { display: flex; flex-wrap: wrap; gap: 8px 18px; padding: 12px 0 10px; }
  .seg { display: flex; gap: 2px; background: var(--surface-sunk); border: 1px solid var(--line); border-radius: 9px; padding: 2px; }
  .seg button {
    border: 0; background: transparent; border-radius: 7px; padding: 5px 11px;
    font-size: .85rem; font-weight: 500; color: var(--ink-soft);
  }
  .seg button[aria-pressed="true"] { background: var(--surface); color: var(--ink); box-shadow: var(--shadow); }
  .seg button:hover:not([aria-pressed="true"]) { color: var(--ink); }

  /* полоса прогресса: клетки, как в GridWorld */
  .rail { display: flex; gap: 3px; padding-bottom: 12px; }
  .rail i {
    flex: 1 1 0; height: 5px; border-radius: 2px; background: var(--line);
    transition: background .18s ease;
  }
  .rail i.seen { background: var(--line-strong); }
  .rail i.ok { background: var(--good); }
  .rail i.no { background: var(--bad); }

  /* ---------- вступление и режимы ---------- */
  .lede { padding: 22px 0 6px; display: grid; gap: 10px; }
  .lede h2 { font-size: 1.5rem; letter-spacing: -.015em; }
  .lede p { color: var(--ink-soft); max-width: 60ch; }
  .note {
    display: flex; gap: 10px; align-items: baseline;
    border-left: 2px solid var(--ice); padding: 2px 0 2px 12px;
    color: var(--ink-soft); font-size: .9rem;
  }
  .note.exam { border-color: var(--amber); }

  /* ---------- вопросы ---------- */
  .q { padding: 26px 0; border-top: 1px solid var(--line); display: grid; gap: 14px; }
  .q:first-of-type { border-top: 0; }
  .q-head { display: grid; gap: 6px; }
  .q-meta { display: flex; align-items: center; gap: 8px; font-size: .74rem; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-soft); }
  .q-meta b { color: var(--ice); font-weight: 600; font-variant-numeric: tabular-nums; }
  .q-text { font-size: 1.06rem; font-weight: 500; text-wrap: pretty; }
  .opts { display: grid; gap: 8px; }
  .opt {
    display: grid; grid-template-columns: 1.6rem 1fr; gap: 10px; align-items: baseline;
    text-align: left; background: var(--surface); border: 1px solid var(--line);
    border-radius: 10px; padding: 11px 14px; transition: border-color .15s ease, background .15s ease;
  }
  .opt:hover:not(:disabled) { border-color: var(--line-strong); }
  .opt:disabled { cursor: default; }
  .opt .key {
    font-family: "JetBrains Mono", ui-monospace, monospace; font-size: .82rem; font-weight: 700;
    color: var(--ink-soft);
  }
  .opt[aria-checked="true"] { border-color: var(--ice); background: var(--ice-soft); }
  .opt[aria-checked="true"] .key { color: var(--ice); }
  .opt.correct { border-color: var(--good); background: var(--good-soft); }
  .opt.correct .key { color: var(--good); }
  .opt.wrong { border-color: var(--bad); background: var(--bad-soft); }
  .opt.wrong .key { color: var(--bad); }
  .why {
    display: grid; gap: 6px; background: var(--surface-sunk); border-radius: 10px;
    padding: 12px 14px; font-size: .93rem;
  }
  .why .verdict { font-weight: 600; }
  .why .verdict.ok { color: var(--good); }
  .why .verdict.no { color: var(--bad); }
  .why .ref { color: var(--ink-soft); font-size: .84rem; }

  /* ---------- зачёт: вход и результат ---------- */
  .gate { display: grid; gap: 14px; max-width: 30rem; padding: 26px 0 8px; }
  label { display: grid; gap: 6px; font-size: .9rem; font-weight: 500; }
  input[type="text"] {
    font: 500 1rem/1.4 "JetBrains Mono", ui-monospace, monospace;
    text-transform: uppercase; letter-spacing: .08em;
    background: var(--surface); color: var(--ink);
    border: 1px solid var(--line-strong); border-radius: 9px; padding: 10px 12px;
  }
  input[type="text"]:focus-visible { outline: 2px solid var(--ice); outline-offset: 1px; }
  .btn {
    border: 1px solid transparent; border-radius: 9px; padding: 10px 18px;
    font-weight: 600; font-size: .94rem; background: var(--ice); color: #fff;
  }
  :root[data-theme="dark"] .btn, :root:not([data-theme="light"]) .btn { color: #07131c; }
  @media (prefers-color-scheme: light) { :root:not([data-theme="dark"]) .btn { color: #fff; } }
  .btn:hover { filter: brightness(1.07); }
  .btn.ghost { background: transparent; color: var(--ink); border-color: var(--line-strong); }
  .btn:disabled { opacity: .45; cursor: default; filter: none; }
  .hint { color: var(--ink-soft); font-size: .86rem; }
  .err { color: var(--bad); font-size: .86rem; min-height: 1.2em; }

  .footbar {
    position: sticky; bottom: 0; z-index: 15;
    margin-inline: -20px; padding: 12px 20px calc(12px + env(safe-area-inset-bottom, 0px));
    background: color-mix(in srgb, var(--paper) 92%, transparent);
    backdrop-filter: blur(8px); border-top: 1px solid var(--line);
    display: flex; align-items: center; justify-content: space-between; gap: 14px; flex-wrap: wrap;
  }
  .footbar .count { font-size: .9rem; color: var(--ink-soft); font-variant-numeric: tabular-nums; }

  .result { display: grid; gap: 16px; padding: 26px 0 8px; }
  .code-panel {
    background: var(--amber-soft); border: 1px solid color-mix(in srgb, var(--amber) 35%, transparent);
    border-radius: 12px; padding: 16px; display: grid; gap: 12px;
  }
  .code-panel .label { font-size: .74rem; letter-spacing: .07em; text-transform: uppercase; color: var(--amber); font-weight: 600; }
  .code {
    font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 1.02rem; font-weight: 700;
    letter-spacing: .04em; word-break: break-all; line-height: 1.45;
  }
  .code-panel .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
  .score { font-family: Literata, Georgia, serif; font-size: 1.35rem; font-weight: 600; }
  .steps { display: grid; gap: 8px; font-size: .93rem; color: var(--ink-soft); padding-left: 1.1rem; margin: 0; }
  .steps code { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: .86em; background: var(--surface-sunk); padding: 1px 5px; border-radius: 4px; }

  footer { border-top: 1px solid var(--line); margin-top: 32px; padding-top: 16px; color: var(--ink-soft); font-size: .84rem; display: grid; gap: 6px; }

  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
  @media (max-width: 480px) {
    .brand h1 { font-size: 1.05rem; }
    .lede h2 { font-size: 1.3rem; }
    .opt { grid-template-columns: 1.4rem 1fr; padding: 10px 12px; }
  }
</style>

<div class="wrap">
  <header>
    <div class="brand">
      <svg class="lake" viewBox="0 0 4 4" aria-hidden="true">__LAKE__</svg>
      <div>
        <h1>Квизы курса RL</h1>
        <div class="sub">Проверка понимания лекций</div>
      </div>
    </div>
    <div class="switches">
      <div class="seg" id="weeks" role="group" aria-label="Неделя"></div>
      <div class="seg" id="modes" role="group" aria-label="Режим">
        <button type="button" id="mode-train" aria-pressed="true">Тренировка</button>
        <button type="button" id="mode-exam" aria-pressed="false">Зачёт</button>
      </div>
    </div>
    <div class="rail" id="rail" aria-hidden="true"></div>
  </header>

  <main id="main"></main>

  <footer>
    <div>Вопросы собраны из лекций курса. Нашли ошибку в формулировке — напишите в <a href="__REPO__/issues">issues репозитория</a>.</div>
    <div>Тренировочный режим ничего не сохраняет и не отправляет. В зачётном режиме страница только собирает ваши ответы в код — проверяет их преподаватель.</div>
  </footer>
</div>

<script>
const QUIZZES = __DATA__;
const LETTERS = "ABCDEFGHIJ";
const SALT = "rl-course-2026";

/* ---------- код результата: crc32 как в zlib ---------- */
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(str) {
  const bytes = new TextEncoder().encode(str);
  let c = 0xFFFFFFFF;
  for (const b of bytes) c = CRC_TABLE[(c ^ b) & 0xFF] ^ (c >>> 8);
  return (c ^ 0xFFFFFFFF) >>> 0;
}

function resultCode(week, student, answers) {
  const letters = answers.map(a => (a === null || a === undefined) ? "X" : LETTERS[a]).join("");
  const payload = "W" + week + ":" + student + ":" + letters;
  const crc = (crc32(payload + SALT) & 0xFFFF).toString(16).toUpperCase().padStart(4, "0");
  return payload + ":" + crc;
}

/* ---------- состояние ---------- */
const store = {
  get(key, fallback) { try { const v = localStorage.getItem(key); return v === null ? fallback : v; } catch (e) { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch (e) { /* приватное окно */ } },
};

let state = {
  week: Number(store.get("rlquiz.week", String(QUIZZES[0].week))),
  mode: store.get("rlquiz.mode", "train") === "exam" ? "exam" : "train",
  student: store.get("rlquiz.student", ""),
  started: false,
  answers: {},
  finished: false,
};

const quiz = () => QUIZZES.find(q => q.week === state.week) || QUIZZES[0];
const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; };

function reset(keepMode) {
  state.answers = {};
  state.finished = false;
  state.started = keepMode ? state.started : false;
}

/* ---------- отрисовка ---------- */
function render() {
  renderSwitches();
  renderRail();
  const main = document.getElementById("main");
  main.replaceChildren();
  if (state.mode === "exam" && !state.started) { main.append(gateView()); return; }
  if (state.mode === "exam" && state.finished) { main.append(resultView()); return; }
  main.append(ledeView());
  const q = quiz();
  q.questions.forEach((item, i) => main.append(questionView(item, i)));
  if (state.mode === "exam") main.append(footbarView());
  else main.append(trainFootView());
}

function renderSwitches() {
  const weeks = document.getElementById("weeks");
  weeks.replaceChildren();
  QUIZZES.forEach(q => {
    const b = el("button", null, "Лекция " + q.week);
    b.type = "button";
    b.setAttribute("aria-pressed", String(q.week === state.week));
    b.onclick = () => { if (q.week === state.week) return; if (!confirmLeave()) return; state.week = q.week; store.set("rlquiz.week", String(q.week)); reset(); render(); };
    weeks.append(b);
  });
  for (const [id, mode] of [["mode-train", "train"], ["mode-exam", "exam"]]) {
    const b = document.getElementById(id);
    b.setAttribute("aria-pressed", String(state.mode === mode));
    b.onclick = () => { if (state.mode === mode) return; if (!confirmLeave()) return; state.mode = mode; store.set("rlquiz.mode", mode); reset(); render(); };
  }
}

function confirmLeave() {
  const answered = Object.keys(state.answers).length;
  if (state.mode !== "exam" || state.finished || answered === 0) return true;
  return confirm("Ответы на этот квиз сбросятся. Продолжить?");
}

function renderRail() {
  const rail = document.getElementById("rail");
  rail.replaceChildren();
  const q = quiz();
  q.questions.forEach((item, i) => {
    const cell = el("i");
    const given = state.answers[i];
    if (given !== undefined) {
      if (state.mode === "train") cell.classList.add(given === item.answer ? "ok" : "no");
      else cell.classList.add("seen");
    }
    rail.append(cell);
  });
}

function ledeView() {
  const q = quiz();
  const box = el("div", "lede");
  box.append(el("h2", null, "Лекция " + q.week + ". " + q.title));
  if (state.mode === "train") {
    box.append(el("p", null, "Двенадцать вопросов по материалу лекции. Выбирайте ответ — разбор появится сразу, попытки не ограничены."));
    const note = el("div", "note");
    note.append(el("span", null, "Ничего не сохраняется и никуда не отправляется: это только для себя."));
    box.append(note);
  } else {
    box.append(el("p", null, "Зачётный режим: правильность ответов не показывается. Отметьте все ответы и нажмите «Завершить» — страница соберёт код результата."));
    const note = el("div", "note exam");
    note.append(el("span", null, "Отвечает " + state.student + ". Ответы можно менять до нажатия «Завершить»."));
    box.append(note);
  }
  return box;
}

function questionView(item, i) {
  const q = quiz();
  const wrap = el("section", "q");
  const head = el("div", "q-head");
  const meta = el("div", "q-meta");
  const num = el("b", null, String(i + 1).padStart(2, "0") + " / " + String(q.questions.length).padStart(2, "0"));
  meta.append(num, el("span", null, item.topic));
  head.append(meta, el("div", "q-text", item.text));
  wrap.append(head);

  const given = state.answers[i];
  const answered = given !== undefined;
  const reveal = state.mode === "train" && answered;

  const opts = el("div", "opts");
  opts.setAttribute("role", "radiogroup");
  opts.setAttribute("aria-label", "Варианты ответа");
  item.options.forEach((text, k) => {
    const b = el("button", "opt");
    b.type = "button";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(given === k));
    b.append(el("span", "key", LETTERS[k]), el("span", null, text));
    if (reveal) {
      b.disabled = true;
      if (k === item.answer) b.classList.add("correct");
      else if (k === given) b.classList.add("wrong");
    }
    b.onclick = () => {
      if (state.mode === "train" && state.answers[i] !== undefined) return;
      state.answers[i] = k;
      render();
      if (state.mode === "train") wrap.scrollIntoView({ block: "nearest", behavior: "smooth" });
    };
    opts.append(b);
  });
  wrap.append(opts);

  if (reveal) {
    const why = el("div", "why");
    const ok = given === item.answer;
    const verdict = el("div", "verdict " + (ok ? "ok" : "no"), ok ? "Верно" : "Неверно, правильный ответ " + LETTERS[item.answer]);
    why.append(verdict, el("div", null, item.explain), el("div", "ref", "Лекция " + q.week + ": " + item.ref));
    wrap.append(why);
  }
  return wrap;
}

function trainFootView() {
  const q = quiz();
  const box = el("div", "result");
  const answered = Object.keys(state.answers).length;
  const right = Object.entries(state.answers).filter(([i, a]) => q.questions[i].answer === a).length;
  if (answered === q.questions.length) {
    box.append(el("div", "score", "Верно " + right + " из " + q.questions.length + (right >= q.pass_score ? " — уверенно" : " — стоит перечитать лекцию")));
  } else if (answered > 0) {
    box.append(el("div", "hint", "Отвечено " + answered + " из " + q.questions.length + ", верно " + right + "."));
  }
  const row = el("div", "code-panel");
  row.style.background = "transparent";
  row.style.border = "0";
  row.style.padding = "0";
  const again = el("button", "btn ghost", "Пройти заново");
  again.type = "button";
  again.onclick = () => { reset(); render(); window.scrollTo({ top: 0, behavior: "smooth" }); };
  const toExam = el("button", "btn", "Перейти к зачёту");
  toExam.type = "button";
  toExam.onclick = () => { state.mode = "exam"; store.set("rlquiz.mode", "exam"); reset(); render(); window.scrollTo({ top: 0 }); };
  const rowBox = el("div", "row");
  rowBox.style.display = "flex";
  rowBox.style.gap = "10px";
  rowBox.style.flexWrap = "wrap";
  rowBox.append(again, toExam);
  box.append(rowBox);
  return box;
}

function gateView() {
  const q = quiz();
  const box = el("form", "gate");
  box.append(el("h2", null, "Зачёт по лекции " + q.week));
  box.append(el("p", "hint", "Введите личный код, который выдал преподаватель, — он войдёт в код результата. Затем ответьте на все двенадцать вопросов: правильность в этом режиме не показывается."));
  const label = el("label", null, "Личный код студента");
  const input = el("input");
  input.type = "text";
  input.id = "student-code";
  input.placeholder = "например IVAN23";
  input.value = state.student;
  input.autocomplete = "off";
  input.spellcheck = false;
  label.append(input);
  const err = el("div", "err");
  err.id = "gate-err";
  const start = el("button", "btn", "Начать");
  start.type = "submit";
  box.append(label, err, start);
  box.onsubmit = (e) => {
    e.preventDefault();
    const code = input.value.trim().toUpperCase();
    if (!/^[A-Z0-9]{3,12}$/.test(code)) {
      err.textContent = "Код состоит из 3–12 латинских букв и цифр — проверьте раскладку.";
      return;
    }
    state.student = code;
    store.set("rlquiz.student", code);
    state.started = true;
    render();
  };
  return box;
}

function footbarView() {
  const q = quiz();
  const bar = el("div", "footbar");
  const answered = Object.keys(state.answers).length;
  bar.append(el("div", "count", "Отвечено " + answered + " из " + q.questions.length));
  const done = el("button", "btn", "Завершить");
  done.type = "button";
  done.disabled = answered === 0;
  done.onclick = () => {
    if (answered < q.questions.length && !confirm("Без ответа осталось " + (q.questions.length - answered) + " вопросов, они пойдут в ноль. Завершить?")) return;
    state.finished = true;
    render();
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  bar.append(done);
  return bar;
}

function resultView() {
  const q = quiz();
  const answers = q.questions.map((_, i) => state.answers[i] === undefined ? null : state.answers[i]);
  const code = resultCode(q.week, state.student, answers);
  const box = el("div", "result");
  box.append(el("h2", null, "Ответы записаны"));
  box.append(el("p", "hint", "Баллы считает преподаватель: страница не знает, какие ответы верные, и никуда их не отправляет. Скопируйте код и сдайте его вместе с домашним заданием."));

  const panel = el("div", "code-panel");
  panel.append(el("div", "label", "Код результата"));
  const codeEl = el("div", "code", code);
  codeEl.id = "result-code";
  panel.append(codeEl);
  const row = el("div", "row");
  const copy = el("button", "btn", "Скопировать");
  copy.type = "button";
  copy.onclick = async () => {
    try {
      await navigator.clipboard.writeText(code);
      copy.textContent = "Скопировано";
      setTimeout(() => { copy.textContent = "Скопировать"; }, 1600);
    } catch (e) {
      const sel = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(codeEl);
      sel.removeAllRanges();
      sel.addRange(range);
      copy.textContent = "Выделено — нажмите Ctrl+C";
    }
  };
  row.append(copy, el("span", "hint", "Код действует только для вашего личного кода " + state.student + "."));
  panel.append(row);
  box.append(panel);

  const steps = el("ol", "steps");
  const li1 = el("li");
  li1.append(document.createTextNode("Откройте свой репозиторий с домашним заданием и файл "));
  li1.append(el("code", null, "quiz/week" + String(q.week).padStart(2, "0") + ".txt"));
  li1.append(document.createTextNode("."));
  const li2 = el("li", null, "Вставьте код одной строкой вместо подсказки и сохраните.");
  const li3 = el("li", null, "Отправьте изменения — проверка в pull request скажет, что код разобрался.");
  steps.append(li1, li2, li3);
  box.append(steps);

  const row2 = el("div", "row");
  row2.style.display = "flex";
  row2.style.gap = "10px";
  row2.style.flexWrap = "wrap";
  const again = el("button", "btn ghost", "Вернуться к тренировке");
  again.type = "button";
  again.onclick = () => { state.mode = "train"; store.set("rlquiz.mode", "train"); reset(); render(); window.scrollTo({ top: 0 }); };
  row2.append(again);
  box.append(row2);
  return box;
}

render();
</script>
"""


def lake_svg() -> str:
    """Карта Frozen Lake 4×4 как знак страницы: S, лёд, проруби, цель."""
    layout = ["SFFF", "FHFH", "FFFH", "HFFG"]
    fills = {"S": "var(--line-strong)", "F": "var(--surface)", "H": "var(--ice)", "G": "var(--good)"}
    cells = []
    for r, row in enumerate(layout):
        for c, ch in enumerate(row):
            cells.append(
                f'<rect x="{c}" y="{r}" width="1" height="1" fill="{fills[ch]}" '
                f'stroke="var(--line)" stroke-width="0.06"/>'
            )
    return "".join(cells)


def build(out: Path) -> None:
    quizzes = []
    for path in sorted(QUIZ_DIR.glob("week*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for q in data["questions"]:
            if not 0 <= q["answer"] < len(q["options"]):
                raise SystemExit(f"{path.name}, {q['id']}: answer вне диапазона вариантов")
        quizzes.append(data)
    if not quizzes:
        raise SystemExit(f"в {QUIZ_DIR} нет файлов week*.json")

    html = (TEMPLATE
            .replace("__DATA__", json.dumps(quizzes, ensure_ascii=False, separators=(",", ":")))
            .replace("__LAKE__", lake_svg())
            .replace("__REPO__", REPO_URL))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    total = sum(len(q["questions"]) for q in quizzes)
    print(f"{out}: {len(quizzes)} квиза, {total} вопросов, {len(html) / 1024:.0f} КБ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Собрать страницу квизов")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    build(parser.parse_args().out)


if __name__ == "__main__":
    main()
