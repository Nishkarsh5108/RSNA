// Builds the supplementary development-history deck from the report's aggregate data tables.
//   npm install pptxgenjs      (once, anywhere; point NODE_PATH at that node_modules folder if it is not beside this file)
//   node build_deck.js         writes Knee_MRI_Development_Notes.pptx next to this script
// Every number on a chart is read from ../data/*.csv; numbers in text are quoted from the report chapters.
// Fonts: Franklin Gothic Medium / Book and Consolas, which ship with Windows and Office.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const DATA = path.join(__dirname, "..", "data");
const OUT = process.env.DECK_OUT || path.join(__dirname, "Knee_MRI_Development_Notes.pptx");

// ------------------------------------------------------------------ data
function csv(name) {
	const text = fs.readFileSync(path.join(DATA, name), "utf8").replace(/^﻿/, "");
	const rows = [];
	let row = [], cell = "", q = false;
	for (let i = 0; i < text.length; i++) {
		const ch = text[i];
		if (q) {
			if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; }
			else if (ch === '"') q = false;
			else cell += ch;
		} else if (ch === '"') q = true;
		else if (ch === ",") { row.push(cell); cell = ""; }
		else if (ch === "\n") { row.push(cell); rows.push(row); row = []; cell = ""; }
		else if (ch !== "\r") cell += ch;
	}
	if (cell !== "" || row.length) { row.push(cell); rows.push(row); }
	const head = rows.shift();
	return rows.filter((r) => r.length > 1).map((r) => Object.fromEntries(head.map((h, i) => [h, r[i]])));
}
const num = (v) => (v === "" || v === undefined ? null : Number(v));
const f3 = (v) => v.toFixed(3);
const sg = (v, d = 3) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);

const subs = csv("submissions_owner.csv").map((r) => ({ n: +r.order, stage: r.stage, what: r.what, pub: num(r.public), priv: num(r.private) }));
const langs = csv("dataset_language.csv").map((r) => ({ lang: r.lang_final, n: +r.studies }));
const labelSrc = csv("label_sources_gold_auc.csv").map((r) => ({ source: r.source, kind: r.kind, v: +r.macro_auc, lo: +r.lo, hi: +r.hi }));
const ablation = csv("ablation_paired_vs_E1.csv").map((r) => ({ run: r.run, v: num(r.pseudo_val_delta), lo: num(r.pseudo_val_lo), hi: num(r.pseudo_val_hi) }));
const curves = csv("curves_mil.csv").map((r) => ({ run: r.run, epoch: +r.epoch, v: +r.pseudo_val_macro_auc }));
const attention = csv("attention_plane_share.csv").map((r) => ({ t: r.target, sag: +r.mass_sag_fs, cor: +r.mass_cor_fs, axi: +r.mass_axi_fs }));
const swap = csv("label_swap_gold.csv").map((r) => ({ model: r.model, labels: r.labels, v: +r.gold_auc, lo: +r.lo, hi: +r.hi, d: num(r.delta_vs_ref), dlo: num(r.d_lo), dhi: num(r.d_hi) }));
const prevRaw = csv("prevalence_by_source.csv");
const prevKeys = Object.keys(prevRaw[0]);
const prev = prevRaw.map((r) => ({ t: r[prevKeys[0]], ours: +r["our report-derived labels"], p7: +r["7th-place image-read labels"], p1: +r["1st-place image-read labels"] }));
const perFinding = csv("gold_per_finding.csv").map((r) => ({ model: r.model, finding: r.finding, v: +r.auc, lo: num(r.lo), hi: num(r.hi) }));
const bpCurves = csv("curves_boxpool.csv").map((r) => ({ run: r.run, epoch: +r.epoch, gold: +r.gold58_macro_auc }));
const summary = JSON.parse(fs.readFileSync(path.join(DATA, "dataset_summary.json"), "utf8"));

// ------------------------------------------------------------------ deck, theme, layouts
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.title = "Knee abnormality detection using deep learning: development notes";
pres.author = "Nishkarsh Singhal, Anuj Chaubey, Vardhan Harsh Nigam";
pres.subject = "Supplementary slides to the live demonstration";
pres.company = "Faculty of Technology, University of Delhi";

const THEME = {
	name: "Knee MRI development notes",
	headFontFace: "Franklin Gothic Medium",
	bodyFontFace: "Franklin Gothic Book",
	colors: {
		dk1: "111111", lt1: "FFFFFF", dk2: "565B63", lt2: "F1F2F3",
		accent1: "F5B700", // signal yellow, the demo site's accent
		accent2: "9AA0A7", // mid grey: secondary marks
		accent3: "D5D8DC", // hairlines, quiet fills
		accent4: "D9482B", // reserved: a result that got worse
		accent5: "FCE7A0", // pale yellow: highlighted rows and bands
		accent6: "30343A", // dark grey
		hlink: "111111", folHlink: "565B63",
	},
};
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const INK = C.text1, GREY = C.text2, WHITE = C.background1, TINT = C.background2;
const YEL = C.accent1, MID = C.accent2, HAIR = C.accent3, RED = C.accent4, PALE = C.accent5, DARK = C.accent6;
const MONO = "Consolas", HEAD = "+mj-lt";

pres.defineSlideMaster({
	title: "COVER",
	objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.92, w: 9.0, h: 1.2, fontSize: 38, color: INK, align: "left", valign: "top", margin: 0 }, text: "" } }],
});
pres.defineSlideMaster({
	title: "CONTENT",
	objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.55, w: 9.0, h: 0.8, fontSize: 21, color: INK, align: "left", valign: "top", margin: 0 }, text: "" } }],
	slideNumber: { x: 8.9, y: 5.2, w: 0.6, h: 0.22, fontFace: MONO, fontSize: 10, color: THEME.colors.dk2, align: "right", margin: 0 },
});

// ------------------------------------------------------------------ drawing helpers
let uid = 0;
const nm = (p) => `${p} ${++uid}`;
function tx(s, text, x, y, w, h, o = {}) {
	s.addText(text, Object.assign({ x, y, w, h, margin: 0, isTextBox: true, fontSize: 11, color: INK, align: "left", valign: "top", objectName: nm("Text") }, o));
}
function mono(s, text, x, y, w, o = {}) {
	tx(s, text, x, y, w, o.h || 0.2, Object.assign({ fontFace: MONO, fontSize: 10, color: GREY, valign: "middle", wrap: false }, o));
}
// single-line label centred vertically on cy
function lab(s, text, x, cy, w, o = {}) {
	tx(s, text, x, cy - 0.11, w, 0.22, Object.assign({ fontSize: 10.5, valign: "middle", wrap: false }, o));
}
function ln(s, x1, y1, x2, y2, o = {}) {
	const line = { color: o.color || INK, width: o.width || 0.75 };
	if (o.dash) line.dashType = o.dash;
	if (o.arrow) line.endArrowType = "triangle";
	s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), flipH: x2 < x1, flipV: y2 < y1, line, objectName: nm("Line") });
}
function rect(s, x, y, w, h, o = {}) {
	const p = { x, y, w, h, objectName: nm(o.name || "Rect") };
	if (o.fill) p.fill = { color: o.fill, transparency: o.transparency || 0 };
	if (o.line) p.line = { color: o.line, width: o.lw || 0.75 };
	if (o.dash) p.line.dashType = o.dash;
	s.addShape(pres.shapes.RECTANGLE, p);
}
function dot(s, cx, cy, r, o = {}) {
	const p = { x: cx - r, y: cy - r, w: 2 * r, h: 2 * r, objectName: nm("Dot") };
	p.fill = { color: o.fill || INK };
	if (o.line) p.line = { color: o.line, width: o.lw || 0.75 };
	s.addShape(pres.shapes.OVAL, p);
}
function poly(s, pts, o = {}) {
	const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
	const x = Math.min(...xs), y = Math.min(...ys);
	const w = Math.max(Math.max(...xs) - x, 0.001), h = Math.max(Math.max(...ys) - y, 0.001);
	const line = { color: o.color || INK, width: o.width || 1.25 };
	if (o.dash) line.dashType = o.dash;
	s.addShape(pres.shapes.CUSTOM_GEOMETRY, { x, y, w, h, line, points: pts.map((p, i) => ({ x: p[0] - x, y: p[1] - y, moveTo: i === 0 })), objectName: nm("Curve") });
}
// a box with text inside (one object, so the text moves with the box)
function box(s, x, y, w, h, runs, o = {}) {
	const p = { x, y, w, h, shape: pres.shapes.RECTANGLE, margin: o.margin || [5, 5, 3, 3], align: o.align || "left", valign: o.valign || "middle", fontSize: 10, color: INK, objectName: nm(o.name || "Block") };
	if (o.fill) p.fill = { color: o.fill };
	p.line = { color: o.line || INK, width: o.lw || 0.75 };
	if (o.dash) p.line.dashType = o.dash;
	if (o.wrap === false) p.wrap = false;
	s.addText(runs, p);
}
const run = (text, o = {}) => ({ text, options: o });
const br = (text, o = {}) => ({ text, options: Object.assign({ breakLine: true }, o) });

// stage tracker: six squares, top right; the slide's stage is filled yellow
function tracker(s, stage) {
	for (let i = 0; i < 6; i++) {
		const x = 9.5 - (6 - i) * 0.17 + 0.07;
		const o = { name: "Stage marker" };
		if (stage === "all" || (typeof stage === "number" && i < stage - 1)) { o.fill = INK; o.line = INK; }
		else if (i === stage - 1) { o.fill = YEL; o.line = INK; }
		else { o.line = MID; }
		rect(s, x, 0.37, 0.1, 0.1, o);
	}
}
function content(section, title, kicker, stage, source) {
	const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
	s.addText(title, { placeholder: "title" });
	mono(s, kicker, 0.5, 0.32, 7.6, { charSpacing: 0.6 });
	tracker(s, stage);
	if (source) mono(s, source, 0.5, 5.21, 8.2);
	return s;
}
// horizontal value axis: baseline, ticks, tick labels
function axisX(s, sx, y, ticks, fmt, o = {}) {
	ln(s, sx(ticks[0]), y, sx(ticks[ticks.length - 1]), y, { color: MID });
	for (const t of ticks) {
		ln(s, sx(t), y, sx(t), y + 0.045, { color: MID });
		tx(s, fmt(t), sx(t) - 0.4, y + 0.06, 0.8, 0.18, { fontFace: MONO, fontSize: 10, color: GREY, align: "center", wrap: false });
	}
	if (o.label) mono(s, o.label, sx(ticks[0]), y + 0.25, sx(ticks[ticks.length - 1]) - sx(ticks[0]), { align: "center" });
}
const scale = (min, max, a, b) => (v) => a + ((v - min) / (max - min)) * (b - a);

// ================================================================== 1  cover
pres.addSection({ title: "Context" });
{
	const s = pres.addSlide({ masterName: "COVER", sectionTitle: "Context" });
	mono(s, "B.TECH PROJECT  ·  ELECTRONICS & COMMUNICATION ENGINEERING  ·  FACULTY OF TECHNOLOGY, UNIVERSITY OF DELHI", 0.5, 0.45, 9, { charSpacing: 0.6 });
	s.addText("Knee abnormality detection\nusing deep learning", { placeholder: "title" });
	tx(s, "Development notes: the experiments, the dead ends and the decisions behind the final model", 0.5, 2.3, 9.0, 0.3, { fontSize: 15, color: GREY, wrap: false });
	mono(s, "SUPPLEMENT TO THE LIVE DEMONSTRATION", 0.5, 2.68, 5, { charSpacing: 0.6 });

	// number line of the private leaderboard score
	const sx = scale(0.45, 1.0, 0.5, 9.5), ay = 4.12;
	mono(s, "PRIVATE LEADERBOARD SCORE  ·  MACRO ROC-AUC OVER TWELVE FINDINGS", 0.5, 3.1, 6, { charSpacing: 0.6 });
	ln(s, 0.5, ay, 9.5, ay, { color: INK, width: 1 });
	for (const t of [0.5, 0.6, 0.7, 0.8, 0.9]) {
		ln(s, sx(t), ay, sx(t), ay + 0.06, { color: INK });
		tx(s, t === 0.5 ? "0.5 chance" : t.toFixed(1), sx(t) - 0.5, ay + 0.09, 1.0, 0.18, { fontFace: MONO, fontSize: 10, color: GREY, align: "center", wrap: false });
	}
	const marks = [
		{ v: 0.498, cap: "first submission", cx: sx(0.498) + 0.25, top: 3.5 },
		{ v: 0.755, cap: "frozen encoder", cx: sx(0.755), top: 3.5 },
		{ v: 0.916, cap: "our own 2.5D model", cx: 7.05, top: 3.5 },
		{ v: 0.943, cap: "competition result", cx: 8.12, top: 3.06 },
		{ v: 0.958, cap: "after the competition", cx: 8.9, top: 3.5, hi: true },
	];
	for (const m of marks) {
		tx(s, m.cap, m.cx - 0.75, m.top, 1.5, 0.18, { fontSize: 10.5, color: GREY, align: "center", wrap: false });
		tx(s, f3(m.v), m.cx - 0.6, m.top + 0.17, 1.2, 0.28, { fontFace: HEAD, fontSize: 16, align: "center", wrap: false });
		ln(s, m.cx, m.top + 0.47, sx(m.v), ay - 0.06, { color: MID });
		dot(s, sx(m.v), ay, 0.06, m.hi ? { fill: YEL, line: INK, lw: 1 } : { fill: INK });
	}
	ln(s, sx(0.96), ay - 0.09, sx(0.96), ay + 0.09, { color: INK, width: 1 });
	tx(s, "winning team 0.960", 8.0, ay + 0.3, 1.5, 0.18, { fontFace: MONO, fontSize: 10, color: GREY, align: "right", wrap: false });
	ln(s, sx(0.96), ay + 0.09, sx(0.96), ay + 0.3, { color: MID });

	mono(s, "TEAM", 0.5, 4.78, 1, { charSpacing: 0.6 });
	tx(s, "Nishkarsh Singhal  ·  Anuj Chaubey  ·  Vardhan Harsh Nigam", 0.5, 4.99, 4.8, 0.22, { fontSize: 12 });
	mono(s, "SUPERVISION", 5.6, 4.78, 2, { charSpacing: 0.6 });
	tx(s, "Dr. Varun Saxena  ·  Dr. Sangeeta Yadav (co-supervisor)", 5.6, 4.99, 3.9, 0.22, { fontSize: 12 });
	s.addNotes(
		"These slides sit beside the demo site. The site explains the final model; these slides explain how we got there.\n" +
		"The number line is the whole story in one row: 0.498 (first submission, a pipeline bug), 0.755 (frozen encoder), 0.916 (our own fine-tuned 2.5D model), 0.943 (our competition result, 490th of 6,299 teams, bronze medal), 0.958 (after the competition, late submissions that are scored but not ranked). The winning team scored 0.960.\n" +
		"All scores are macro ROC-AUC over the twelve findings on the hidden test set (about 1,300 studies)."
	);
}

// ================================================================== 2  the task
{
	const s = content("Context", "Only 58 of 4,407 training studies carry expert labels; the rest come with a free-text report", "CONTEXT  ·  THE TASK AND ITS DATA", null, "REPORT §1.1  ·  TABLES 1.1–1.2  ·  FIG. 1.1");
	// waffle: one square = 10 studies
	const x0 = 0.5, y0 = 1.6, pitch = 0.142, size = 0.116, n = 21;
	for (let r = 0; r < n; r++)
		for (let c = 0; c < n; c++) {
			const gold = r === 0 && c >= n - 6;
			rect(s, x0 + c * pitch, y0 + r * pitch, size, size, gold ? { fill: YEL, line: INK, lw: 0.75, name: "Gold studies" } : { fill: HAIR, name: "Report-only studies" });
		}
	mono(s, "1 SQUARE ≈ 10 STUDIES", x0, y0 + n * pitch + 0.03, 3);

	const mx = 3.85;
	ln(s, x0 + n * pitch - 0.01, y0 + size / 2, mx - 0.1, y0 + size / 2, { color: INK });
	tx(s, String(summary.gold), mx, 1.42, 2.0, 0.62, { fontFace: HEAD, fontSize: 44, wrap: false });
	tx(s, "studies read by experts for all twelve findings. We call them gold-58.", mx, 2.1, 1.95, 0.6, { fontSize: 12 });
	tx(s, (summary.studies - summary.gold).toLocaleString("en-US"), mx, 3.0, 2.2, 0.62, { fontFace: HEAD, fontSize: 44, wrap: false });
	tx(s, "studies with nothing but a radiology report, written in one of nine languages.", mx, 3.68, 1.95, 0.6, { fontSize: 12 });

	// report languages
	const rx = 6.3;
	mono(s, "REPORT LANGUAGE  ·  STUDIES", rx, 1.58, 3.2, { charSpacing: 0.6 });
	const bw = scale(0, 1720, 0, 1.65);
	langs.forEach((l, i) => {
		const cy = 1.93 + i * 0.205;
		lab(s, l.lang, rx, cy, 0.85, { align: "right" });
		rect(s, rx + 0.95, cy - 0.06, bw(l.n), 0.12, { fill: INK, name: "Language bar" });
		lab(s, l.n.toLocaleString("en-US"), rx + 0.95 + bw(l.n) + 0.07, cy, 0.6, { fontFace: MONO, fontSize: 10, color: GREY });
	});
	mono(s, "AT TEST TIME", rx, 3.93, 3.2, { charSpacing: 0.6 });
	const facts = ["Images only: no report, no label", "12 probabilities per study; macro ROC-AUC", "\u2248 1,300 hidden studies; 9 h offline, one GPU"];
	facts.forEach((f, i) => tx(s, f, rx, 4.18 + i * 0.25, 3.2, 0.22, { fontSize: 11.5, wrap: false }));
	s.addNotes(
		"The task: read a complete knee MRI study (raw DICOM, three planes, median 5 series and 162 slices) and output a probability for each of twelve findings. Metric: macro ROC-AUC, so a rare finding such as MCL injury counts as much as effusion.\n" +
		"The real difficulty is supervision. 4,407 training studies, only 58 with expert labels for all twelve findings (gold-58). The other 4,349 have only a free-text report in one of nine languages (English 1,720, Spanish 682, Turkish 546, Croatian 406, Greek 321, German 262, Bulgarian 220, Dutch 154, French 96).\n" +
		"Reports are not available at test time, so they can only be used to make training labels. The model must work from images alone.\n" +
		"Gold-58 is also unusual: every gold study has at least one positive finding (4.1 on average), so it contains no normal knees. With 58 studies, one study moves a finding's AUC by about 0.01.\n" +
		"Constraints: code competition, 9 hours offline on one Kaggle T4 for about 1,300 hidden studies; our own hardware was one laptop with a 6 GB GPU."
	);
}

// ================================================================== 3  path
{
	const s = content("Context", "Six stages, each chosen by what limited the result of the one before", "OVERVIEW  ·  THE DEVELOPMENT PATH", null, "REPORT §2.5.1, FIG. 2.2  ·  FIG. 3.6  ·  APPENDIX B (ALL 22 SUBMISSIONS)");
	const segs = [
		{ id: 1, w: 3.0, subs: [1, 2], name: "frozen features" },
		{ id: 2, w: 3.3, subs: [], name: "audit, new labeller" },
		{ id: 3, w: 3.0, subs: [3, 4], name: "own 2.5D model" },
		{ id: 4, w: 12.0, subs: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], name: "ensembles built around public models" },
		{ id: 5, w: 6.6, subs: [17, 18, 19, 20, 21, 22], name: "published design, fused labels" },
		{ id: 6, w: 3.3, subs: [], name: "local workstation" },
	];
	const gap = 0.3, X0 = 1.0, X1 = 9.5, top = 2.25, bot = 4.62;
	const u = (X1 - X0) / (segs.reduce((a, g) => a + g.w, 0) + gap * (segs.length - 1));
	const sy = scale(0.48, 0.98, bot, top);
	const pos = {};
	let x = X0;
	for (const g of segs) {
		g.x = x; g.px = g.w * u;
		g.subs.forEach((n, i) => (pos[n] = g.x + ((i + 0.5) / g.subs.length) * g.px));
		x += g.px + gap * u;
	}
	for (const g of segs) {
		if (g.subs.length) rect(s, g.x, top, g.px, bot - top, { fill: TINT, name: "Stage band" });
		mono(s, "STAGE " + g.id, g.x, 1.56, g.px, { color: INK, charSpacing: 0.6 });
		tx(s, g.name, g.x, 1.76, g.px, 0.4, { fontSize: 10.5, color: GREY });
	}
	ln(s, X0 - 0.06, top, X0 - 0.06, bot, { color: MID });
	for (const t of [0.5, 0.6, 0.7, 0.8, 0.9]) {
		ln(s, X0 - 0.11, sy(t), X0 - 0.06, sy(t), { color: MID });
		lab(s, t.toFixed(1), 0.42, sy(t), 0.42, { fontFace: MONO, fontSize: 10, color: GREY, align: "right" });
	}
	ln(s, X0, sy(0.96), X1, sy(0.96), { color: INK, dash: "dash" });
	const g6 = segs[5];
	tx(s, "winning team", g6.x, sy(0.96) + 0.05, g6.px, 0.17, { fontSize: 10.5, color: GREY, wrap: false });
	tx(s, "0.960", g6.x, sy(0.96) + 0.21, g6.px, 0.2, { fontFace: HEAD, fontSize: 11.5, wrap: false });

	const scored = subs.filter((r) => r.priv !== null);
	let chain = [];
	for (const r of subs) {
		if (r.priv === null) { if (chain.length > 1) poly(s, chain, { color: MID, width: 0.75 }); chain = []; continue; }
		chain.push([pos[r.n], sy(r.priv)]);
	}
	if (chain.length > 1) poly(s, chain, { color: MID, width: 0.75 });
	for (const r of scored) dot(s, pos[r.n], sy(r.priv), 0.05, r.n === 9 || r.n === 17 ? { fill: YEL, line: INK, lw: 1 } : { fill: INK });

	const note = (val, text, nx, ny, w) => {
		tx(s, [run(val, { fontFace: HEAD, fontSize: 11.5, breakLine: true }), run(text, { fontSize: 10.5, color: GREY })], nx, ny, w, 0.6, {});
	};
	note(f3(subs[0].priv), "a borrowed image cache did not hold the test studies", pos[1] + 0.16, sy(0.498) - 0.46, 2.3);
	note(f3(subs[1].priv), "own DICOM pipeline", pos[2] + 0.12, sy(0.755) - 0.1, 1.4);
	note(f3(subs[2].priv), "fine-tuned encoder", pos[3] - 0.12, sy(0.916) + 0.1, 0.9);
	note(f3(subs[8].priv), "final pick: 490th of 6,299 teams, bronze medal", pos[5] - 0.05, sy(0.943) + 0.14, 1.75);
	note("0.850–0.935", "last day: blend weights fitted on the 58 gold studies, one run stopped at its time limit. None selected.", pos[8] - 0.1, sy(0.76), 2.25);
	ln(s, pos[14] - 0.05, sy(0.772), pos[15] - 0.04, sy(0.842), { color: MID });
	note(f3(subs[16].priv), "late submissions: scored, not ranked", pos[17] - 0.05, sy(0.958) + 0.14, 1.55);
	mono(s, "ONE DOT = ONE SUBMISSION, IN ORDER  ·  PRIVATE LEADERBOARD MACRO AUC  ·  SUBMISSION 12 RETURNED NO SCORE", X0, 4.74, 8.5);
	s.addNotes(
		"How to read it: every dot is one of our 22 submissions, in the order made, with its private-leaderboard score. Bands are the stages.\n" +
		"Stage 1: frozen ImageNet encoder. First submission 0.498 (chance) because a third-party image cache did not contain the hidden test studies; with our own DICOM pipeline 0.755.\n" +
		"Stage 2: no submissions. We audited the data and labels and rebuilt the report labeller.\n" +
		"Stage 3: our own fine-tuned 2.5D multiple-instance model, 0.916 private (0.927 public).\n" +
		"Stage 4: final week of the competition. Ensembles built around the strongest public models, each re-run and checked by us: 0.937, 0.940, 0.943. Final result 0.943, 490th of 6,299 teams, bronze. The dip on the last day is real and we show it: blends whose weights were fitted on the 58 gold studies and that ran close to the 9-hour limit scored 0.850 to 0.935; none was selected.\n" +
		"Stage 5: after the competition. Controlled label-swap experiment, then the published first-place design retrained on fused labels: 0.958 as late submissions (scored by Kaggle, not ranked). Winning team: 0.960.\n" +
		"Stage 6: the local workstation, which is the demo.\n" +
		"Working rules: the decision rule was written down before each test, and every decision, including rejected ones, went into a decision log."
	);
}

// ================================================================== 4  stage 1 architecture
pres.addSection({ title: "Stages 1 and 2" });
{
	const s = content("Stages 1 and 2", "The first model kept its encoder frozen: 0.775 once the images came from our own DICOM pipeline", "STAGE 1  ·  FIRST LABELS AND FROZEN-FEATURE BASELINES", 1, "REPORT §3.3.1  ·  APPENDIX B  ·  src/model/knee_mri_v1–v4");
	mono(s, "ARCHITECTURE OF VERSIONS 1 AND 2", 0.5, 1.5, 5, { charSpacing: 0.6 });
	const by = 1.76, bh = 0.86;
	const blocks = [
		{ x: 0.5, w: 1.5, t: "Input", d: ["3 planes × 12 slices", "one series per plane", "224 × 224 px"] },
		{ x: 2.3, w: 1.75, t: "EfficientNet-B0", d: ["ImageNet weights", "frozen, features cached", "1,280-d per slice"] },
		{ x: 4.35, w: 1.85, t: "Attention pooling", d: ["one vector per plane", "1280→128, tanh, →1", "softmax over 12 slices"] },
		{ x: 6.5, w: 1.3, t: "Concatenate", d: ["3 planes", "3 × 1,280", "= 3,840-d"] },
		{ x: 8.1, w: 1.4, t: "MLP head", d: ["3840→256→12", "ReLU, dropout 0.3", "12 logits"] },
	];
	blocks.forEach((b, i) => {
		box(s, b.x, by, b.w, bh, [br(b.t, { fontFace: HEAD, fontSize: 11.5 }), ...b.d.map((d, j) => (j < b.d.length - 1 ? br(d, { fontSize: 10, color: GREY }) : run(d, { fontSize: 10, color: GREY })))], { fill: b.fill, valign: "middle" });
		if (i) ln(s, blocks[i - 1].x + blocks[i - 1].w + 0.03, by + bh / 2, b.x - 0.03, by + bh / 2, { arrow: true });
	});
	tx(s, [run("Labels  ", { fontFace: HEAD }), run("Qwen2.5-7B (vLLM on 2 × T4) writes JSON probabilities; consensus with public label sets.   ", { color: GREY }), run("Loss  ", { fontFace: HEAD }), run("weighted BCE.   ", { color: GREY }), run("Speed  ", { fontFace: HEAD }), run("cached features, 441 s → 5 s per epoch.", { color: GREY })], 0.5, 2.7, 9, 0.38, { fontSize: 10.5 });

	// versions table with public leaderboard bars from chance
	const ty = 3.26, rh = 0.36;
	mono(s, "VERSION", 0.5, ty - 0.02, 1, { charSpacing: 0.6 });
	mono(s, "WHAT CHANGED", 1.3, ty - 0.02, 3, { charSpacing: 0.6 });
	mono(s, "PUBLIC LEADERBOARD AUC", 6.3, ty - 0.02, 3.2, { charSpacing: 0.6 });
	const bx = scale(0.5, 0.8, 6.3, 9.05);
	const rows = [
		{ v: "v1", t: "Images read from a preprocessed archive shared by another participant", auc: subs[0].pub, note: "0.500: the archive did not hold the test studies" },
		{ v: "v2", t: "Own DICOM pipeline: slices ordered along the scan normal, 1–99% window", auc: subs[1].pub, hi: true },
		{ v: "v3", t: "+ right-knee mirroring, up to 3 series per plane, region-guided heads", none: "no better on validation, not submitted" },
		{ v: "v4", t: "Frozen DINOv2-S encoder (384-d) in place of EfficientNet-B0", none: "no better on validation, not submitted" },
	];
	rows.forEach((r, i) => {
		const y = ty + 0.24 + i * rh;
		ln(s, 0.5, y, 9.5, y, { color: HAIR });
		tx(s, r.v, 0.5, y + 0.07, 0.6, 0.26, { fontFace: HEAD, fontSize: 12, valign: "middle" });
		tx(s, r.t, 1.3, y + 0.03, 4.75, 0.34, { fontSize: 10.5, valign: "middle" });
		const cy = y + rh / 2;
		if (r.auc !== undefined) {
			ln(s, bx(0.5), cy - 0.11, bx(0.5), cy + 0.11, { color: INK, width: 1 });
			if (r.auc > 0.5) rect(s, bx(0.5), cy - 0.07, bx(r.auc) - bx(0.5), 0.14, r.hi ? { fill: YEL, line: INK } : { fill: INK });
			if (r.note) lab(s, r.note, bx(0.5) + 0.1, cy, 3.1, { color: GREY });
			else lab(s, f3(r.auc), bx(r.auc) + 0.08, cy, 0.6, { fontFace: HEAD, fontSize: 11.5 });
		} else lab(s, r.none, bx(0.5), cy, 3.2, { color: GREY });
	});
	const yb = ty + 0.24 + rows.length * rh;
	ln(s, 0.5, yb, 9.5, yb, { color: HAIR });
	mono(s, "0.5 = CHANCE", bx(0.5) - 0.02, yb + 0.03, 1.2);
	mono(s, "0.8", bx(0.8) - 0.3, yb + 0.03, 0.6, { align: "center" });
	ln(s, bx(0.8), yb, bx(0.8), yb + 0.04, { color: MID });
	s.addNotes(
		"Before this we read the knee MRI literature (MRNet: per-slice 2D CNN, max over slices, logistic regression over planes) and ran two public MRNet implementations to understand the per-slice-encoder-plus-pooling design. Our versions 1 to 4 follow that family, with attention pooling instead of max.\n" +
		"Versions 1 and 2: twelve slices from one series per plane; each slice through an ImageNet-pretrained EfficientNet-B0 that stays frozen (1,280 features per slice); one attention pooling layer shared by the planes (Linear 1280 to 128, tanh, Linear to 1, softmax over slices); the three plane vectors are concatenated (3,840) and go through a small MLP (256 hidden, dropout 0.3) to twelve logits. Because the encoder is frozen its features are cached: 441 s per epoch became about 5 s.\n" +
		"Labels at this stage: Qwen2.5-7B-Instruct served with vLLM on Kaggle's two T4s (the 14B model ran out of memory), asked for a JSON object with twelve probabilities, report cut to 1,200 characters; combined with public label sets by a consensus procedure.\n" +
		"v1 scored 0.500 public: it read images from a third-party preprocessed archive that did not include the hidden test studies, so the notebook had no real images. v2 replaced it with our own DICOM pipeline (slice order from ImagePositionPatient projected on the slice normal, 1st to 99th percentile intensity window) and scored 0.775 public / 0.755 private.\n" +
		"v3 added mirroring of right knees, up to three series per plane and region-guided heads: five anatomical groups (ligaments, menisci, OA compartments, fluid, trauma), each with its own attention over the three planes. v4 swapped in a frozen DINOv2-small encoder. Neither did better on our validation.\n" +
		"What we did not know yet: all four versions picked their best epoch by AUC on gold studies and trained on gold with 8x weight, so their validation numbers were optimistic. The next stage found this."
	);
}

// ================================================================== 5  audit
{
	const s = content("Stages 1 and 2", "An audit found nine defects in the first pipeline, several of them silent; its reported scores were optimistic", "STAGE 2  ·  DATA AND LABEL AUDIT", 2, "REPORT §3.3.2, TABLE 3.1  ·  AUDIT OF 819,078 DICOM HEADERS, ALL REPORTS, EVERY LABEL FILE");
	const rows = [
		["52,188", "label cells in one public pseudo-label file were exactly 0.5; with a ≥ 0.5 rule it voted “positive” everywhere", "corrupted the label consensus"],
		["1,744", "cells of our first LLM labels were exactly 0.5, and a failed JSON parse would write 0.5 for all twelve findings, unflagged", "uncertain counted as positive"],
		["37.7%", "of reports were cut by the 1,200-character limit, often before the conclusion", "findings missed"],
		["95%", "of Greek reports carried the micro sign in place of the letter mu", "text unreadable to the labeller"],
		["≈ half", "of the right knees were left unmirrored, because only the exact tag “R” was handled", "medial and lateral mixed up"],
		["50.6 / 17.3%", "ACL-positive rate when a series is named “ACL”, against all other studies", "a shortcut the model could learn"],
		["58", "gold studies chose the best epoch and the consensus weights, and were also trained on", "optimistic validation scores", true],
	];
	const y0 = 1.56, rh = 0.435;
	mono(s, "MEASURED", 0.5, y0 - 0.05, 1.5, { charSpacing: 0.6 });
	mono(s, "DEFECT", 2.3, y0 - 0.05, 3, { charSpacing: 0.6 });
	mono(s, "CONSEQUENCE", 7.05, y0 - 0.05, 2.4, { charSpacing: 0.6 });
	rows.forEach((r, i) => {
		const y = y0 + 0.2 + i * rh;
		if (r[3]) rect(s, 0.5, y, 9.0, rh, { fill: PALE, name: "Highlighted row" });
		ln(s, 0.5, y, 9.5, y, { color: HAIR });
		tx(s, r[0], 0.6, y, 1.65, rh, { fontFace: HEAD, fontSize: 18, valign: "middle", wrap: false });
		tx(s, r[1], 2.3, y, 4.55, rh, { fontSize: 11, valign: "middle" });
		tx(s, r[2], 7.05, y, 2.4, rh, { fontSize: 11, color: r[3] ? INK : GREY, valign: "middle" });
	});
	const yb = y0 + 0.2 + rows.length * rh;
	ln(s, 0.5, yb, 9.5, yb, { color: HAIR });
	tx(s, "Also found: undecodable planes were zero-filled, and no JPEG-Lossless decoder was installed for the possibly compressed test set.", 0.5, yb + 0.08, 9, 0.22, { fontSize: 10.5, color: GREY, wrap: false });
	s.addNotes(
		"Why we stopped to audit: the jump from 0.500 to 0.775 came from a bug fix, and we had no reliable way to tell whether later changes helped. We read all 819,078 DICOM headers, all reports and every label file.\n" +
		"Table 3.1 of the report lists nine defects; the slide shows seven and names the other two in the footnote. Several were silent: the pipeline ran without errors and produced plausible numbers.\n" +
		"Constant voter: one 'public pseudo-label' file held exactly 0.5 for all 52,188 unlabelled cells, so with a >= 0.5 rule it voted positive for everything in the consensus.\n" +
		"Silent fallback: when the LLM's JSON failed to parse, the script wrote 0.5 for all twelve findings with no flag; 1,744 cells in the final 7B file were exactly 0.5.\n" +
		"Truncation: the 1,200-character cut shortened 37.7% of all reports and half of the gold reports, often removing the conclusion.\n" +
		"Greek: 95% of Greek reports used the micro sign (U+00B5) instead of the letter mu.\n" +
		"Laterality: only studies tagged exactly 'R' were mirrored, about half of the right knees. The tag is present for only 50% of studies; we later recovered the side from patient coordinates (97.8% agreement with the tag).\n" +
		"Protocol shortcut: studies with a series named 'ACL' were ACL-positive in 50.6% of reports against 17.3% otherwise.\n" +
		"Gold leakage into selection: consensus weights and best epochs were chosen on gold, and gold rows were trained on. From here on gold-58 was never trained on, and for our own model it was sealed and scored once.\n" +
		"Two decoding defects complete the nine: planes that failed to decode became blank images, and no JPEG-Lossless decoder was installed."
	);
}

// ================================================================== 6  labeller
{
	const s = content("Stages 1 and 2", "A forced-choice labeller that never writes free text raised agreement with expert labels from 0.80 to 0.87", "STAGE 2  ·  THE NEW REPORT LABELLER", 2, "REPORT §2.5.3, FIG. 2.3  ·  §3.3.2, FIG. 3.1b, TABLE 3.2");
	// left: how it works
	mono(s, "HOW IT READS A REPORT (SCHEMATIC)", 0.5, 1.5, 3.9, { charSpacing: 0.6 });
	rect(s, 0.5, 1.76, 2.05, 1.3, { line: INK, name: "Answer sheet" });
	const sheet = ["ACL injury: D", "MCL injury: D", "Medial meniscus: _", "Lateral meniscus:", "Medial OA:", "…"];
	sheet.forEach((t, i) => {
		const y = 1.82 + i * 0.2;
		if (i === 2) rect(s, 0.56, y + 0.005, 1.93, 0.19, { fill: YEL, name: "Current blank" });
		tx(s, t, 0.62, y, 1.9, 0.2, { fontFace: MONO, fontSize: 10, color: i > 2 ? GREY : INK, valign: "middle", wrap: false });
	});
	ln(s, 2.58, 2.31, 2.86, 2.31, { arrow: true });
	// schematic next-token distribution
	const hx = 2.98, hb = 2.82;
	const hs = [0.62, 0.2, 0.05, 0.08, 0.05];
	["A", "B", "C", "D", "E"].forEach((L, i) => {
		const x = hx + i * 0.2;
		rect(s, x, hb - hs[i] * 0.92, 0.13, hs[i] * 0.92, { fill: i === 0 ? INK : MID, name: "Schematic bar" });
		tx(s, L, x - 0.035, hb + 0.03, 0.2, 0.18, { fontFace: MONO, fontSize: 10, align: "center", wrap: false });
	});
	ln(s, hx - 0.05, hb, hx + 0.98, hb, { color: MID });
	tx(s, "next-token probabilities at the blank", hx - 0.05, 1.76, 1.62, 0.36, { fontSize: 10, color: GREY });
	tx(s, [run("A", { fontFace: MONO }), run(" present   "), run("B", { fontFace: MONO }), run(" possible   "), run("C", { fontFace: MONO }), run(" not mentioned   "), run("D", { fontFace: MONO }), run(" absent   "), run("E", { fontFace: MONO }), run(" prior only")], 0.5, 3.13, 4.0, 0.2, { fontSize: 10, color: GREY, wrap: false });
	tx(s, "φ = p(A) + 0.5 p(B) + 0.25 p(E)", 0.5, 3.42, 3.8, 0.3, { fontFace: HEAD, fontSize: 15, wrap: false });
	const steps = [
		"Nothing is parsed, so nothing can fail silently",
		"Two passes with the items in opposite order, averaged",
		"Training label = mean with two public label sets",
		"Local Qwen3-4B on a 6 GB GPU: 4,407 reports, 0 failures",
	];
	steps.forEach((t, i) => {
		const y = 3.86 + i * 0.25;
		rect(s, 0.5, y + 0.065, 0.07, 0.07, { fill: INK, name: "Marker" });
		tx(s, t, 0.68, y, 3.6, 0.2, { fontSize: 10.5, valign: "middle", wrap: false });
	});

	// right top: label sources against gold-58
	const keep = ["Rule extractor (9 languages)", "Qwen2.5-7B (first pipeline)", "Public report labels P1", "Our labeller (Qwen3-4B), one order", "Our labeller, two orders averaged", "Public report labels P2", "Training labels (ours + P1 + P2)"];
	const names = { "Rule extractor (9 languages)": "Rule extractor, 9 languages", "Qwen2.5-7B (first pipeline)": "First labeller, Qwen2.5-7B", "Public report labels P1": "Public label set P1", "Our labeller (Qwen3-4B), one order": "New labeller, one item order", "Our labeller, two orders averaged": "New labeller, two orders", "Public report labels P2": "Public label set P2", "Training labels (ours + P1 + P2)": "Training labels: ours + P1 + P2" };
	const hi = new Set(["Our labeller, two orders averaged", "Training labels (ours + P1 + P2)"]);
	const src = keep.map((k) => labelSrc.find((r) => r.source === k));
	const LX = 4.75, PX0 = 7.1, PX1 = 9.0;
	mono(s, "AGREEMENT WITH GOLD-58  ·  MACRO AUC, 95% CI", LX, 1.5, 4.7, { charSpacing: 0.6 });
	const sx = scale(0.74, 0.96, PX0, PX1);
	src.forEach((r, i) => {
		const cy = 1.86 + i * 0.215;
		if (hi.has(r.source)) rect(s, LX - 0.05, cy - 0.105, 9.5 - LX + 0.05, 0.21, { fill: PALE, name: "Highlighted row" });
		lab(s, names[r.source], LX, cy, 2.3, hi.has(r.source) ? { fontFace: HEAD } : {});
		ln(s, sx(r.lo), cy, sx(r.hi), cy, { color: INK, width: 1 });
		dot(s, sx(r.v), cy, 0.045, hi.has(r.source) ? { fill: YEL, line: INK, lw: 1 } : { fill: INK });
		lab(s, f3(r.v), 9.05, cy, 0.45, { fontFace: MONO, fontSize: 10, align: "right" });
	});
	axisX(s, sx, 1.86 + 7 * 0.215 - 0.06, [0.75, 0.8, 0.85, 0.9, 0.95], (t) => t.toFixed(2));

	// right bottom: does a better label set train a better frozen model?
	mono(s, "FROZEN MODEL RETRAINED ON EACH LABEL SET  ·  GOLD-58 AUC", LX, 3.82, 4.7, { charSpacing: 0.6 });
	const s2 = scale(0.5, 0.8, LX + 0.1, 9.3), ay = 4.5;
	rect(s, s2(0.51), ay - 0.07, s2(0.59) - s2(0.51), 0.14, { fill: HAIR, name: "Gold-only range" });
	tx(s, "58 gold studies alone", s2(0.51) - 0.3, ay - 0.42, 1.6, 0.18, { fontSize: 10.5, color: GREY, wrap: false });
	tx(s, "0.51–0.59", s2(0.51) - 0.3, ay - 0.26, 1.6, 0.18, { fontFace: MONO, fontSize: 10, wrap: false });
	[0.747, 0.758, 0.761, 0.772, 0.782].forEach((v) => dot(s, s2(v), ay, 0.045, v === 0.782 ? { fill: YEL, line: INK, lw: 1 } : { fill: INK }));
	tx(s, "five report-derived label sets", s2(0.765) - 1.45, ay - 0.42, 2.0, 0.18, { fontSize: 10.5, color: GREY, align: "right", wrap: false });
	tx(s, "0.747–0.782", s2(0.765) - 1.45, ay - 0.26, 2.0, 0.18, { fontFace: MONO, fontSize: 10, align: "right", wrap: false });
	axisX(s, s2, ay + 0.12, [0.5, 0.6, 0.7, 0.8], (t) => t.toFixed(1));
	tx(s, "Label sets differ by about 0.03: the frozen image model was the limit", LX, 4.94, 4.75, 0.2, { fontFace: HEAD, fontSize: 10.5, wrap: false });
	s.addNotes(
		"The first labeller asked the LLM to write JSON and parsed it with regular expressions. The redesign removes free-text output completely.\n" +
		"How it works: the full report (after Unicode normalisation and the Greek repair) goes into a prompt that defines the twelve findings and five answer letters: A present, B possible, C not mentioned, D absent, E prior only. The answer layout is fixed ('ACL injury: _', ...). At each blank we read the model's next-token distribution over the five letters directly, then write the most likely letter into the blank and move on, exactly as greedy decoding would. Nothing is parsed, so there is no failure fallback. The bar chart on the slide is a schematic, not measured data.\n" +
		"Score: phi = p(A) + 0.5 p(B) + 0.25 p(E), weights fixed before any gold evaluation. Because earlier answers can influence later ones, we run two passes with the items in opposite order and average.\n" +
		"Model: Qwen3-4B-Instruct in 4-bit NF4 on the 6 GB laptop GPU. Two engineering fixes made it practical: asking for logits at the last position only (the default needed 3.6 GB for an 8,000-token prompt), and caching the key-value state of the shared system prompt (about 12x faster). 0.94 to 1.6 studies/s, 4.4 GB peak, about 75 minutes per pass, all 4,407 reports with no failure; probability mass on the five letters at least 0.995 everywhere. The prompt was developed on 60 non-gold studies and frozen before its first gold evaluation.\n" +
		"Chart, agreement with gold-58: first labeller 0.802; new labeller one order 0.857 (paired gain +0.055, 95% CI +0.012 to +0.093); two orders averaged 0.868. Final training labels = mean of ours and two public report-label sets P1 and P2: 0.890 [0.853, 0.922]. Caveat we state openly: six of the 58 gold rows in P1 equal the gold labels, so 0.890 may be slightly optimistic. The rule extractor (0.791) was a second opinion only and was not used for training.\n" +
		"Bottom chart, the decision it led to: the frozen model retrained from scratch on each label set (fixed epochs, five seeds, gold only for the final score) scored 0.747 to 0.782, against 0.51 to 0.59 when trained on the 58 gold studies alone. So the labels are clearly useful, but label sets differ by only about 0.03, which 58 studies cannot resolve. With frozen features the image model was the limit, so Stage 3 worked on the image model."
	);
}

// ================================================================== 7  2.5D MIL architecture
pres.addSection({ title: "Stage 3" });
{
	const s = content("Stage 3", "Fine-tuning the encoder inside a 2.5D attention network was the largest gain from our own model: +0.090", "STAGE 3  ·  OUR 2.5D MULTIPLE-INSTANCE MODEL", 3, "REPORT §2.5.4, FIG. 2.4  ·  §3.3.3, TABLE 3.3, FIG. 3.2a");
	mono(s, "ARCHITECTURE  ·  28.4 M PARAMETERS", 0.5, 1.5, 4.4, { charSpacing: 0.6 });
	const bx = 0.5, bw = 1.72, bh = 0.4, gp = 0.135, y0 = 1.76;
	const blocks = [
		{ t: "Input windows", a: ["3 series × 8 windows × 3 slices, 224 px", "neighbouring slices ≈ 3.5 mm apart"] },
		{ t: "ConvNeXt-T encoder", a: ["shared by all 24 windows, fine-tuned", "27.8 M parameters → 768-d"], hi: true },
		{ t: "Projection", a: ["Linear 768→384, LayerNorm", "+ plane embedding + position MLP"] },
		{ t: "12 attention heads", a: ["α = softmax(W₂ tanh(W₁t + b₁))", "over 24 tokens, one per finding"] },
		{ t: "12 linear heads", a: ["z = ⟨Σ α·t, v⟩ + b, then sigmoid", "one output per finding"] },
	];
	blocks.forEach((b, i) => {
		const y = y0 + i * (bh + gp);
		box(s, bx, y, bw, bh, [run(b.t, { fontFace: HEAD, fontSize: 11.5 })], { fill: b.hi ? YEL : undefined, valign: "middle" });
		tx(s, [br(b.a[0], { fontSize: 10 }), run(b.a[1], { fontSize: 10, color: GREY })], bx + bw + 0.14, y - 0.01, 2.8, bh + 0.02, { valign: "middle" });
		if (i < blocks.length - 1) ln(s, bx + bw / 2, y + bh + 0.01, bx + bw / 2, y + bh + gp - 0.01, { arrow: true });
	});
	const yr = y0 + blocks.length * (bh + gp) + 0.02;
	tx(s, [run("Trained on the laptop  ", { fontFace: HEAD }), run("soft-label BCE, AdamW, cosine schedule, weight EMA 0.999; 8 epochs in 57 min at 3.8 GB.", { color: GREY })], 0.5, yr, 4.5, 0.4, { fontSize: 10.5 });

	// right: validation curves
	const PX0 = 5.75, PX1 = 9.0, PT = 1.86, PB = 4.2;
	mono(s, "MACRO AUC ON 870 HELD-OUT STUDIES, BY EPOCH", 5.3, 1.5, 4.2, { charSpacing: 0.6 });
	const ex = scale(1, 20, PX0, PX1), ey = scale(0.72, 0.9, PB, PT);
	for (const t of [0.75, 0.8, 0.85, 0.9]) {
		ln(s, PX0, ey(t), PX1, ey(t), { color: HAIR });
		lab(s, t.toFixed(2), PX0 - 0.5, ey(t), 0.42, { fontFace: MONO, fontSize: 10, color: GREY, align: "right" });
	}
	ln(s, PX0, PB, PX1, PB, { color: MID });
	for (const t of [1, 4, 8, 12, 16, 20]) {
		ln(s, ex(t), PB, ex(t), PB + 0.045, { color: MID });
		tx(s, String(t), ex(t) - 0.3, PB + 0.06, 0.6, 0.18, { fontFace: MONO, fontSize: 10, color: GREY, align: "center", wrap: false });
	}
	mono(s, "EPOCH", PX0, PB + 0.26, PX1 - PX0, { align: "center" });
	const e0 = curves.filter((r) => r.run === "E0 frozen features"), e1 = curves.filter((r) => r.run === "E1");
	poly(s, e0.map((r) => [ex(r.epoch), ey(r.v)]), { color: MID, width: 1.75 });
	poly(s, e1.map((r) => [ex(r.epoch), ey(r.v)]), { color: INK, width: 2.25 });
	const e1end = e1[e1.length - 1], e0best = e0.reduce((a, b) => (b.v > a.v ? b : a));
	dot(s, ex(e1end.epoch), ey(e1end.v), 0.055, { fill: YEL, line: INK, lw: 1 });
	tx(s, [br("fine-tuned encoder", { fontFace: HEAD, fontSize: 11 }), run(f3(e1end.v) + " after 8 epochs", { fontFace: MONO, fontSize: 10 })], ex(8) + 0.14, ey(e1end.v) - 0.2, 2.0, 0.4, {});
	tx(s, [br("frozen encoder, mean of 3 seeds", { fontFace: HEAD, fontSize: 11, color: GREY }), run("peaks at " + f3(e0best.v) + ", then declines", { fontFace: MONO, fontSize: 10, color: GREY })], ex(9.2), ey(0.775) + 0.1, 2.5, 0.4, {});
	// paired gain bracket
	const gx = ex(8) - 0.02;
	ln(s, gx, ey(e1end.v) + 0.08, gx, ey(0.792) - 0.02, { color: INK, width: 1 });
	ln(s, gx - 0.05, ey(0.792) - 0.02, gx + 0.05, ey(0.792) - 0.02, { color: INK, width: 1 });
	tx(s, [br("+0.090", { fontFace: HEAD, fontSize: 15 }), run("paired, 95% CI 0.079–0.101", { fontSize: 10, color: GREY })], ex(8) + 0.14, ey(0.845) - 0.05, 2.2, 0.45, {});
	tx(s, [run("Gold-58, scored once, 3 seeds: ", { color: GREY }), run("0.753 → 0.906", { fontFace: HEAD })], 5.3, 4.78, 4.2, 0.22, { fontSize: 10.5, wrap: false });
	s.addNotes(
		"This is the model we designed and trained ourselves, entirely on the laptop.\n" +
		"Input: three series per study (sagittal, coronal, axial; fluid-sensitive first, fall back to the other sequence, mask a missing plane). Each series is covered by 8 windows spread over 4 to 96% of the stack; a window stacks three slices as the three colour channels, with the step chosen from the slice spacing so that channels are about 3.5 mm apart. That is the '2.5D' input. Slices are ordered along the slice normal, intensities mapped with the series' own 1st and 99th percentiles (vendors differ by a factor of 25), right knees mirrored.\n" +
		"Network: every window goes through one shared ConvNeXt-Tiny (ImageNet-22k pretrained, from timm), fine-tuned end to end. The 768-d feature is projected to 384 with dropout 0.2 and LayerNorm; a learned plane embedding and a small MLP of the slice position are added. That gives 24 tokens. Each of the twelve findings has its own attention distribution over the tokens (Ilse et al. style: W1 is 256x384, W2 is 12x256) and its own linear head. 28.4 M parameters, 27.8 M of them in the encoder.\n" +
		"What the algebra says: the study logit is an attention-weighted average of per-window scores. A finding seen in one window raises the logit only if attention concentrates there, and evidence from two planes is averaged, not combined with a logical 'and'.\n" +
		"Training: binary cross-entropy against the soft labels; AdamW (1e-4 encoder, 5e-4 head, weight decay 0.05), half-epoch warm-up then cosine, gradient clipping 1.0, effective batch of 4 studies, bfloat16, EMA 0.999 (the EMA weights are the ones evaluated and shipped). Augmentation: one affine per series, gamma, gain, random window centres; no left-right flips because they would undo the laterality normalisation. 8 epochs on 3,479 studies took about 57 minutes at 3.8 GB.\n" +
		"Result: 0.882 [0.873, 0.891] on our held-out fold of 870 studies against 0.792 for the frozen design re-run under the same protocol (best of three seeds; the grey curve is their mean). Paired gain +0.090 [0.079, 0.101]; all twelve findings improved. On gold-58, scored once after the choice was fixed: 0.906 [0.877, 0.932] for the three-seed ensemble against 0.753.\n" +
		"Note the held-out fold is scored against report-derived labels, so it measures agreement with report labels."
	);
}

// ================================================================== 8  ablations
{
	const s = content("Stage 3", "Seven further design changes were tested against a rule fixed in advance; none cleared seed-to-seed noise", "STAGE 3  ·  ABLATIONS AND A SANITY CHECK", 3, "REPORT §3.3.3, FIG. 3.3a, FIG. 3.4a  ·  PAIRED BOOTSTRAP, 2,000 RESAMPLES");
	const A = (k) => ablation.find((r) => r.run.startsWith(k));
	const rows = [
		{ k: "E1s2", l: "Same recipe, second seed", ref: true },
		{ k: "E2c", l: "Transformer layer over tokens" },
		{ k: "E1L", l: "12 epochs instead of 8" },
		{ k: "E8", l: "Finer 0.6 mm pixel grid" },
		{ k: "E4", l: "Six series instead of three" },
		{ k: "E1p", l: "Slice-position bug fix (kept)" },
		{ k: "E5w", l: "Confidence-weighted labels" },
		{ k: "E2a", l: "Mean pooling, no attention" },
		{ k: "E6", l: "EfficientNetV2-S encoder" },
	];
	mono(s, "CHANGE IN MACRO AUC VS BASE RECIPE  ·  870 STUDIES  ·  95% CI", 0.5, 1.5, 5.6, { charSpacing: 0.6 });
	const PX0 = 2.95, PX1 = 5.35, y0 = 1.9, rh = 0.262;
	const sx = scale(-0.022, 0.01, PX0, PX1);
	const yb = y0 + rows.length * rh - 0.03;
	rect(s, sx(-0.003), y0 - 0.14, sx(0.003) - sx(-0.003), yb - y0 + 0.14, { fill: PALE, name: "Noise band" });
	ln(s, sx(0), y0 - 0.14, sx(0), yb, { color: INK, width: 0.75 });
	rows.forEach((r, i) => {
		const a = A(r.k), cy = y0 + i * rh;
		lab(s, r.l, 0.5, cy, 2.35, r.ref ? { color: GREY } : {});
		ln(s, sx(a.lo), cy, sx(a.hi), cy, { color: INK, width: 1 });
		dot(s, sx(a.v), cy, 0.045, r.ref ? { fill: WHITE, line: INK, lw: 1 } : { fill: INK });
		lab(s, sg(a.v), PX1 + 0.08, cy, 0.6, { fontFace: MONO, fontSize: 10 });
	});
	axisX(s, sx, yb, [-0.02, -0.01, 0, 0.01], (t) => (t === 0 ? "0" : sg(t, 2)));
	const fr = A("E0");
	tx(s, [run("Off the scale: ", { color: GREY }), run("frozen encoder " + sg(fr.v), { fontFace: HEAD }), run(".  Shaded: ± the two-seed difference, 0.003.", { color: GREY })], 0.5, yb + 0.36, 5.5, 0.2, { fontSize: 10.5, wrap: false });

	// right: attention share by plane
	const HX = 6.25, CW = 0.62, LW = 1.3, hy = 1.97, hr = 0.2;
	mono(s, "WHERE EACH FINDING'S ATTENTION GOES", HX, 1.5, 3.3, { charSpacing: 0.6 });
	["sagittal", "coronal", "axial"].forEach((p, j) => tx(s, p, HX + LW + j * CW, hy - 0.19, CW, 0.18, { fontSize: 10.5, color: GREY, align: "center", wrap: false }));
	attention.forEach((r, i) => {
		const y = hy + i * hr;
		lab(s, r.t, HX, y + hr / 2, LW - 0.08, {});
		[r.sag, r.cor, r.axi].forEach((v, j) => {
			rect(s, HX + LW + j * CW + 0.012, y + 0.012, CW - 0.024, hr - 0.024, { fill: YEL, transparency: Math.max(0, Math.round(100 - (v / 0.61) * 100)), name: "Attention cell" });
			tx(s, Math.round(v * 100) + "%", HX + LW + j * CW, y, CW, hr, { fontFace: MONO, fontSize: 10, align: "center", valign: "middle", wrap: false });
		});
	});
	tx(s, "Share of attention per plane, 300 validation studies. The model was never told where a finding is.", HX, hy + 12 * hr + 0.06, 3.25, 0.36, { fontSize: 10.5, color: GREY });
	tx(s, [run("Submitted: 3 seeds + shifted-window averaging  ", { color: GREY }), run("0.927 public, 0.916 private", { fontFace: HEAD })], 0.5, 4.86, 5.5, 0.22, { fontSize: 10.5, wrap: false });
	s.addNotes(
		"After the fine-tuning gain we tested one change at a time against the base recipe (run E1). The decision rule was written down before each result was seen: a positive point estimate with a lower confidence bound at or above zero on held-out data.\n" +
		"The yardstick is the first row: the unchanged recipe with another seed differs by +0.003 on the held-out fold (and +0.013 on gold-58). The shaded band is plus/minus that seed difference.\n" +
		"Inside the noise: a transformer layer over the tokens (+0.002 at 1.17x the cost), 12 epochs (+0.001), a finer 0.6 mm pixel grid (-0.002), six series instead of three (-0.003).\n" +
		"Clearly worse: confidence-weighted labels (-0.005), mean pooling instead of attention (-0.008), EfficientNetV2-S encoder (-0.015). On gold-58 all three pointed the same way.\n" +
		"The slice-position fix (-0.003) is a correctness fix found by the attention analysis: the position input was mirrored for right-knee sagittal series. We kept it because it is correct, not because of the score.\n" +
		"So the design was kept as it was. Later, re-scored against image-read labels, the seed spread was about 0.007 and the same verdicts held.\n" +
		"Right panel, a sanity check on what the model learned: each finding has its own attention distribution, so we can see where it looks without ever having told it. MCL puts half its weight on the coronal plane (collateral ligaments are seen along their length there); lateral meniscus 60% sagittal; effusion and Baker's cyst prefer axial. In the sagittal plane the lateral-meniscus head looks at the lateral end of the stack (mean position 0.39) and the medial-meniscus head at the medial end (0.64), which can only happen if the laterality normalisation works.\n" +
		"Test-time averaging of two window sets (second shifted by half a step): +0.0017 [0.0001, 0.0033], adopted. Final package: three seeds trained on all 4,349 report-labelled studies. Gold-58 0.906; leaderboard 0.927 public / 0.916 private. Inference 1.9 s per study, about 42 minutes for the test set, 70 to 80% of it DICOM decoding.\n" +
		"A higher-resolution fourth model gained +0.005 on the fold but scored 0.928 / 0.915 on the leaderboard: no real change."
	);
}

// ================================================================== 9  endgame
pres.addSection({ title: "Stage 4" });
{
	const s = content("Stage 4", "In the final week we engineered ensembles around the strongest public models: 0.916 to 0.943 private", "STAGE 4  ·  THE COMPETITION ENDGAME", 4, "REPORT §3.3.4, TABLES 3.4–3.5  ·  PUBLIC MODELS BY NARTAA AND HELIOSLI (KAGGLE), RE-RUN LOCALLY BY US");
	// left top: public models
	mono(s, "PUBLIC MODELS, EACH RE-RUN ON OUR LAPTOP", 0.5, 1.5, 5.3, { charSpacing: 0.6 });
	mono(s, "GOLD-58", 4.3, 1.5, 0.75, { align: "right" });
	mono(s, "S / STUDY", 5.1, 1.5, 0.85, { align: "right" });
	const models = [
		["CoAtNet", "nartaa", "CoAtNet at 384 px; trained with 2,399 extra knees from the OAI", "0.923", "≈ 3"],
		["S6", "heliosli", "ConvNeXt with an anatomy-crop model trained on SKM-TEA", "0.927", "19"],
		["s6fall", "nartaa", "S6 retrained on all report-labelled studies", "0.931", "6.5"],
		["R6", "heliosli", "ConvNeXt at 384 px, same crop model; auxiliary task on fastMRI+", "0.937", "54–62"],
	];
	const my = 1.74, mh = 0.4;
	models.forEach((m, i) => {
		const y = my + i * mh;
		rect(s, 0.5, y + 0.03, 1.02, mh - 0.06, { fill: HAIR, name: "Public model" });
		tx(s, [br(m[0], { fontFace: HEAD, fontSize: 11.5 }), run(m[1], { fontFace: MONO, fontSize: 10, color: GREY })], 0.57, y + 0.03, 0.95, mh - 0.06, { valign: "middle" });
		tx(s, m[2], 1.64, y, 2.7, mh, { fontSize: 10.5, valign: "middle" });
		tx(s, m[3], 4.3, y, 0.75, mh, { fontFace: MONO, fontSize: 10.5, align: "right", valign: "middle", wrap: false });
		tx(s, m[4], 5.1, y, 0.85, mh, { fontFace: MONO, fontSize: 10.5, align: "right", valign: "middle", wrap: false });
		ln(s, 0.5, y + mh, 5.95, y + mh, { color: HAIR });
	});
	// left bottom: our work
	mono(s, "OUR CHECKS AND ENGINEERING", 0.5, 3.5, 5.3, { charSpacing: 0.6 });
	const ours = [
		["3 datasets", "traced behind the public models; one pick kept fastMRI-free"],
		["57 of 58", "studies identical after s6fall moved inside R6’s decode pass"],
		["0 of 6", "blend upgrades passed the rule fixed in advance"],
		["0.928 / 0.931", "our fine-tune of s6fall against its start: not adopted"],
	];
	ours.forEach((r, i) => {
		const y = 3.74 + i * 0.3;
		tx(s, r[0], 0.5, y, 1.35, 0.28, { fontFace: HEAD, fontSize: 12.5, valign: "middle", wrap: false });
		tx(s, r[1], 1.9, y, 4.1, 0.28, { fontSize: 10.5, valign: "middle", wrap: false });
	});

	// right top: the two final picks
	const RX = 6.4, RW = 3.1;
	mono(s, "OUR TWO FINAL PICKS", RX, 1.5, RW, { charSpacing: 0.6 });
	tx(s, [run("final-strict", { fontFace: HEAD, fontSize: 11.5 }), run("   fastMRI-free, weights fixed in advance", { fontSize: 10, color: GREY })], RX, 1.76, RW, 0.2, { wrap: false });
	const seg = [["CoAtNet", 0.5], ["S6", 0.25], ["s6fall", 0.25]];
	let px = RX;
	seg.forEach(([n, w]) => {
		box(s, px, 2.0, RW * w, 0.34, [run(n + " " + w.toFixed(2), { fontSize: 10 })], { fill: HAIR, line: WHITE, lw: 1.5, align: "center", margin: 0, wrap: false });
		px += RW * w;
	});
	tx(s, [run("0.956", { fontFace: HEAD }), run(" public   ", { color: GREY }), run("0.940", { fontFace: HEAD }), run(" private", { color: GREY })], RX, 2.39, RW, 0.2, { fontSize: 11, wrap: false });

	tx(s, [run("final-max", { fontFace: HEAD, fontSize: 11.5 }), run("   a third model inside one decode pass", { fontSize: 10, color: GREY })], RX, 2.78, RW, 0.2, { wrap: false });
	const seg2 = [["CoAtNet", 1 / 3, HAIR], ["R6", 1 / 3, HAIR], ["s6fall, in-pass", 1 / 3, YEL]];
	px = RX;
	seg2.forEach(([n, w, f]) => {
		box(s, px, 3.02, RW * w, 0.34, [run(n, { fontSize: 10 })], { fill: f, line: WHITE, lw: 1.5, align: "center", margin: 0, wrap: false });
		px += RW * w;
	});
	tx(s, [run("0.958", { fontFace: HEAD }), run(" public   ", { color: GREY }), run("0.943", { fontFace: HEAD }), run(" private   490th of 6,299", { color: GREY })], RX, 3.41, RW, 0.2, { fontSize: 11, wrap: false });

	// right bottom: the lesson
	mono(s, "LAST DAY: WEIGHTS FIT ON GOLD-58", RX, 3.86, RW, { charSpacing: 0.6 });
	const lx0 = RX + 0.95, lx1 = RX + 1.8, vy = scale(0.93, 0.956, 4.9, 4.3);
	ln(s, lx0, vy(0.939), lx1, vy(0.954), { color: INK, width: 1.25 });
	ln(s, lx0, vy(0.94), lx1, vy(0.935), { color: RED, width: 1.25 });
	dot(s, lx0, vy(0.9395), 0.045, { fill: INK });
	dot(s, lx1, vy(0.954), 0.045, { fill: INK });
	dot(s, lx1, vy(0.935), 0.045, { fill: RED });
	tx(s, [br("before", { fontSize: 10, color: GREY }), run("0.939 / 0.940", { fontFace: MONO, fontSize: 10 })], RX - 0.25, vy(0.9395) - 0.17, 1.1, 0.34, { align: "right", valign: "middle" });
	tx(s, [br("0.954", { fontFace: HEAD, fontSize: 11 }), run("gold-58, in-sample", { fontSize: 10, color: GREY })], lx1 + 0.1, vy(0.954) - 0.17, 1.3, 0.34, { valign: "middle" });
	tx(s, [br("0.935", { fontFace: HEAD, fontSize: 11, color: RED }), run("private score", { fontSize: 10, color: RED })], lx1 + 0.1, vy(0.935) - 0.17, 1.3, 0.34, { valign: "middle" });
	s.addNotes(
		"Situation with five days left: the best public notebooks scored 0.954 to 0.958 on the public leaderboard against our 0.927, and every public notebook above 0.950 used a CoAtNet that its author had trained with 2,399 additional knees from the OAI. Our Stage 3 ablations had found nothing that moved our model beyond noise. Closing 0.03 with one model of our own in five days was not realistic, so we changed strategy: build ensembles around the strongest public models and spend our effort on checking, combining and re-engineering them. The models are cited on the slide; the ensembles and the engineering are ours.\n" +
		"The public models (Table 3.4): CoAtNet at 384 px by nartaa, trained with OAI data; S6 by heliosli, a ConvNeXt with an anatomy-crop model trained on SKM-TEA; s6fall by nartaa, S6 retrained on all report-labelled studies; R6 by heliosli, ConvNeXt at 384 px with the same crop model and an auxiliary task on fastMRI+ (inferred from its trainer). Gold-58 scores and seconds per study are from our own local runs with the authors' code.\n" +
		"Provenance: external data was allowed only if freely and equally available. OAI needs only registration and a click-through agreement; fastMRI+ needs an institutional affiliation. So one of our two final picks was kept free of fastMRI-derived models.\n" +
		"Running every model ourselves needed porting. One fast DICOM reader used os.pread, which does not exist on Windows; the error was swallowed and every volume came back empty with no message. We patched it and verified byte-identical output.\n" +
		"Blends: members and weights were fixed in writing before each submission was scored. final-strict = CoAtNet 0.5, S6 0.25, s6fall 0.25; we predicted 0.955 to 0.956 public and it scored 0.956. Six further blend upgrades (standardised logits, per-compartment weights, mirrored test-time averaging, a calibration adapter, ...) were tested on gold-58 with a rule fixed in advance: none passed. Mirror averaging lowered S6 by 0.0066 [-0.0107, -0.0026] on 300 held-out studies.\n" +
		"Fitting one more model into nine hours: the strongest public notebook already needed 6.5 to 8 hours, so a model added as a separate stage could not finish. We rewrote the inference worker so that s6fall scores a second, 320-pixel rendering inside R6's existing decode pass. It matched the separate stage to 2e-10 on the first gold study and on 57 of 58 studies; the one difference was a study whose knee side was decided differently.\n" +
		"Our own fine-tunes of s6fall on the 6 GB GPU (150 GB slice cache, frozen early stages, gradient checkpointing, 2.3 GB peak, BCE plus a pairwise ranking term): 0.928 then 0.937 on gold-58 against 0.931; in the blends the gain was +0.0007 to +0.0010 with intervals crossing zero. Not adopted.\n" +
		"Result: final-max v2 0.958 public / 0.943 private, 490th of 6,299 teams, bronze medal. The pick that scored 0.943 contains R6. The fastMRI-free pick scored 0.940.\n" +
		"The lesson at bottom right: on the last day we relaxed our own rule and fitted per-finding blend weights on the 58 gold studies. Gold went from 0.939 to 0.954, but that was in-sample; on the private leaderboard the fastMRI-free version scored 0.935, below the 0.940 of the blend it extended. Other last-day runs scored 0.850 to 0.894; one is known to have hit its internal 8.3-hour deadline. None was selected."
	);
}

// ================================================================== 10  label swap
pres.addSection({ title: "Stage 5" });
{
	const s = content("Stage 5", "Changing only the label file lifted our model from 0.891 to 0.928: labels, not design, were the bottleneck", "STAGE 5  ·  POST-COMPETITION DIAGNOSIS", 5, "REPORT §3.3.5, FIG. 3.1a, FIG. 3.7  ·  IMAGE-READ LABELS PUBLISHED BY THE 1ST- AND 7TH-PLACED TEAMS");
	// left: prevalence, ours vs image-read
	mono(s, "SHARE OF STUDIES CALLED POSITIVE, 4,349 STUDIES", 0.5, 1.5, 4.3, { charSpacing: 0.6 });
	const order = prev.slice().sort((a, b) => b.ours - (b.p1 + b.p7) / 2 - (a.ours - (a.p1 + a.p7) / 2));
	const hiSet = new Set(["MCL", "Lateral OA", "Medial OA", "PF OA", "Effusion"]);
	const PX0 = 1.95, PX1 = 4.45, y0 = 2.05, rh = 0.208;
	const sx = scale(0, 0.6, PX0, PX1);
	dot(s, 0.56, 1.8, 0.045, { fill: INK });
	lab(s, "our report labels", 0.68, 1.8, 1.4, { fontSize: 10.5 });
	rect(s, 2.05, 1.76, 0.3, 0.08, { fill: MID, name: "Legend" });
	lab(s, "two image-read label sets", 2.42, 1.8, 2.0, { fontSize: 10.5 });
	order.forEach((r, i) => {
		const cy = y0 + i * rh;
		if (hiSet.has(r.t)) rect(s, 0.45, cy - rh / 2 + 0.005, PX1 - 0.45 + 0.1, rh - 0.01, { fill: PALE, name: "Highlighted row" });
		lab(s, r.t, 0.5, cy, 1.4, hiSet.has(r.t) ? { fontFace: HEAD } : {});
		const a = Math.min(r.p1, r.p7), b = Math.max(r.p1, r.p7);
		ln(s, sx((a + b) / 2), cy, sx(r.ours), cy, { color: MID, width: 0.75 });
		rect(s, sx(a), cy - 0.04, Math.max(sx(b) - sx(a), 0.03), 0.08, { fill: MID, name: "Image-read range" });
		dot(s, sx(r.ours), cy, 0.045, { fill: INK });
	});
	axisX(s, sx, y0 + 12 * rh - 0.06, [0, 0.2, 0.4, 0.6], (t) => Math.round(t * 100) + "%");
	tx(s, "Highlighted: findings with a size or grade threshold, over-called in reports", 0.5, 4.84, 4.5, 0.2, { fontSize: 10.5, color: GREY, wrap: false });

	// right: label swap forest
	const LX = 5.15, QX0 = 7.3, QX1 = 9.0;
	mono(s, "GOLD-58 MACRO AUC BY TRAINING LABELS, 95% CI", LX, 1.5, 4.35, { charSpacing: 0.6 });
	const qx = scale(0.86, 0.98, QX0, QX1);
	const mil = swap.filter((r) => r.model.startsWith("2.5D")), bp = swap.filter((r) => r.model.startsWith("Box"));
	const lbl = { "Our report-derived labels": "our report labels", "1st-place labels": "1st-place image-read", "7th-place labels": "7th-place image-read", "Fused 1st + 7th": "fused: mean of both" };
	let y = 1.82;
	const group = (title, sub, rows, hiLabel) => {
		tx(s, [run(title, { fontFace: HEAD, fontSize: 11.5 }), run("  " + sub, { fontSize: 10, color: GREY })], LX, y, 4.35, 0.2, { wrap: false });
		y += 0.31;
		rows.forEach((r) => {
			const h = r.labels === hiLabel;
			if (h) rect(s, LX - 0.05, y - 0.105, 9.5 - LX + 0.05, 0.21, { fill: PALE, name: "Highlighted row" });
			lab(s, lbl[r.labels], LX + 0.12, y, 2.0, h ? { fontFace: HEAD } : {});
			ln(s, qx(r.lo), y, qx(r.hi), y, { color: INK, width: 1 });
			dot(s, qx(r.v), y, 0.045, h ? { fill: YEL, line: INK, lw: 1 } : { fill: INK });
			lab(s, f3(r.v), 9.05, y, 0.45, { fontFace: MONO, fontSize: 10, align: "right" });
			y += 0.215;
		});
	};
	group("Our 2.5D model", "same code, fold and seed", mil, "Fused 1st + 7th");
	y += 0.12;
	group("Published box-pooling model", "same recipe", bp, null);
	axisX(s, qx, y - 0.02, [0.86, 0.9, 0.94, 0.98], (t) => t.toFixed(2));
	const d7 = mil.find((r) => r.labels === "7th-place labels");
	tx(s, [run(sg(d7.d) + " ", { fontFace: HEAD, fontSize: 15 }), run("our model, 7th-place labels; 95% CI " + sg(d7.dlo) + " to " + sg(d7.dhi), { fontSize: 10.5, color: GREY })], LX, 4.52, 4.35, 0.26, { valign: "middle", wrap: false });
	tx(s, [run("Synovitis alone: ", { color: GREY }), run("0.689 → 0.857", { fontFace: HEAD }), run(". The published model is a tie.", { color: GREY })], LX, 4.8, 4.35, 0.22, { fontSize: 10.5, wrap: false });
	s.addNotes(
		"After the competition closed the top teams published their methods; the first- and seventh-placed teams released their label sets, and the first-placed team its code. We read twelve write-ups. They agreed on four points: labels made by LLM agents that look at the images as well as the report score 0.93 to 0.95 against gold-58 (ours: 0.890); pooling each finding inside an anatomical region adds a further gain; large ensembles add little; a single well-built model reaches 0.954 to 0.958 private.\n" +
		"That gave us a testable hypothesis for why our own model stopped at 0.916.\n" +
		"Left chart, measuring our labels: on the same 4,349 studies our report labels call MCL injury in 14.7% against 2 to 3% for the image readers, lateral OA 23.5% against 5 to 8%, PF OA 44.6% against 19 to 22%, effusion 59.4% against 31 to 34%. Reports mention small effusions and mild degenerative change that an image reader working to a threshold calls negative. The opposite holds for synovitis: reports rarely mention it (12%), gold readers mark it in 47% of their studies. No report reader can fix either problem.\n" +
		"Right chart, the label-swap experiment: we retrained our exact Stage 3 recipe (same fold, same seed, same code) and changed only the label file. Report labels 0.891; first-place labels 0.912 (+0.021, CI +0.004 to +0.039); seventh-place labels 0.927 (+0.036, CI +0.018 to +0.055); mean of the two 0.928. Synovitis alone rose from 0.689 to 0.857. Confirmed on our held-out fold, each model scored against the image-read set it was not trained on: +0.029 and +0.026.\n" +
		"This answers the question left open in Stage 2: once the encoder was fine-tuned, the labels limited our model more than its design did. It also means our earlier 'no gain' verdicts were judged with a noisy ruler, so we re-scored the Stage 3 ablations against image-read labels: seed spread about 0.007; higher resolution, six series, twelve epochs and the transformer head all inside it.\n" +
		"Lower group: the published box-pooling model trained on the same three label choices is a tie (0.9388, 0.9389, 0.9383). The stronger architecture is much less sensitive to the label source. That is why we fused the two label sets rather than choosing one.\n" +
		"Caveats: one seed per run; the first-place labels were partly fitted to gold, so their own gold agreement is in-sample."
	);
}

// ================================================================== 11  final architecture
{
	const s = content("Stage 5", "We reproduced the published first-place pipeline (0.955 vs 0.954), then trained its classifier on fused labels: 0.958", "STAGE 5  ·  THE IMPROVED MODEL", 5, "REPORT §2.5.5, FIG. 2.5  ·  §3.3.5, TABLES 3.6–3.7  ·  DESIGN: TEAM KANGAROO (1ST PLACE), OPEN SOURCE");
	const by = 1.74, bh = 1.42;
	mono(s, "PIPELINE", 0.5, 1.5, 2, { charSpacing: 0.6 });
	rect(s, 6.02, 1.555, 0.16, 0.1, { fill: HAIR, line: MID, name: "Legend" });
	mono(s, "PUBLISHED DESIGN", 6.25, 1.5, 1.5, {});
	rect(s, 8.07, 1.555, 0.16, 0.1, { fill: PALE, line: INK, name: "Legend" });
	mono(s, "OUR ADDITIONS", 8.3, 1.5, 1.2, {});
	const B = [
		{ x: 0.5, w: 0.82, t: "Study", d: ["all series", "raw DICOM"], fill: null },
		{ x: 1.56, w: 1.72, t: "Side classifier", d: ["ResNet-18d", "coronal + axial slices", "224 px, ≤ 64 slices", "mirror so every knee", "reads as a right knee"], fill: HAIR, tag: "PUBLISHED WEIGHTS" },
		{ x: 3.52, w: 2.0, t: "ROI detector", d: ["ConvNeXt-T per slice, 160 px", "4 stages fused in 3D", "FiLM: plane, side, fat-sat, FOV", "one 3D box per finding", "+ mirrored pass, averaged"], fill: HAIR, tag: "PUBLISHED WEIGHTS" },
		{ x: 5.76, w: 2.2, t: "Box-pooling classifier", d: ["112 mm crop at 256 px", "ConvNeXt-T → 8 × 8 × 768 cells", "weight = box share × e^attention", "one linear head per finding", "training only: lesion + report heads"], fill: HAIR, tag: "TRAINED BY US, SIX RUNS" },
		{ x: 8.2, w: 1.3, t: "Average", d: ["over series,", "then over the", "trained models", "→ 12 probabilities"], fill: PALE },
	];
	B.forEach((b, i) => {
		box(s, b.x, by, b.w, bh, [br(b.t, { fontFace: HEAD, fontSize: 11.5 }), ...b.d.map((d, j) => (j < b.d.length - 1 || b.tag ? br(d, { fontSize: 10 }) : run(d, { fontSize: 10 }))), ...(b.tag ? [run(b.tag, { fontFace: MONO, fontSize: 10, color: GREY })] : [])], { fill: b.fill || undefined, line: b.fill === HAIR ? MID : INK, valign: "top", margin: [5, 4, 4, 5] });
		if (i) ln(s, B[i - 1].x + B[i - 1].w + 0.03, by + 0.3, b.x - 0.03, by + 0.3, { arrow: true });
	});
	// our layer
	const oy = 3.38, oh = 0.34;
	box(s, 5.76, oy, 2.2, oh, [run("Fused labels: ½ (y", { fontSize: 10.5 }), run("1st", { fontSize: 10.5, subscript: true }), run(" + y", { fontSize: 10.5 }), run("7th", { fontSize: 10.5, subscript: true }), run(")", { fontSize: 10.5 })], { fill: PALE, align: "center" });
	ln(s, 6.86, oy - 0.01, 6.86, by + bh + 0.02, { arrow: true });
	box(s, 3.52, oy, 2.0, oh, [run("Six training runs, folds 0–3", { fontSize: 10.5 })], { fill: PALE, align: "center" });
	box(s, 0.5, oy, 2.78, oh, [run("Inference script rebuilt, no TensorRT", { fontSize: 10.5 })], { fill: PALE, align: "center" });
	box(s, 8.2, oy, 1.3, oh, [run("Fixed in advance", { fontSize: 10.5 })], { fill: PALE, align: "center" });

	// bottom left: six runs on gold-58
	mono(s, "GOLD-58 AUC AT THE LAST EPOCH", 0.5, 3.94, 4.6, { charSpacing: 0.6 });
	const runs = ["A0", "A1", "A2", "C1", "C2", "C3"].map((k) => {
		const rr = bpCurves.filter((r) => r.run.startsWith(k));
		return { k, v: rr[rr.length - 1].gold };
	});
	const six = perFinding.find((r) => r.model.startsWith("Box-pooling") && r.finding === "macro").v;
	const gx = scale(0.934, 0.946, 0.85, 4.55), gy = 4.76;
	ln(s, gx(0.934), gy, gx(0.946), gy, { color: MID });
	for (const t of [0.934, 0.938, 0.942, 0.946]) {
		ln(s, gx(t), gy, gx(t), gy + 0.045, { color: MID });
		tx(s, f3(t), gx(t) - 0.4, gy + 0.06, 0.8, 0.18, { fontFace: MONO, fontSize: 10, color: GREY, align: "center", wrap: false });
	}
	const R = Object.fromEntries(runs.map((r) => [r.k, r.v]));
	runs.forEach((r) => dot(s, gx(r.v), gy, 0.045, { fill: INK }));
	const tag = (text, x, lift) => tx(s, text, x - 0.5, gy - lift - 0.17, 1.0, 0.17, { fontFace: MONO, fontSize: 10, align: "center", wrap: false });
	tag("C1", gx(R.C1), 0.06);
	tag("A2", gx(R.A2) - 0.08, 0.06);
	tag("A0 A1", gx((R.A0 + R.A1) / 2), 0.26);
	ln(s, gx((R.A0 + R.A1) / 2), gy - 0.26, gx((R.A0 + R.A1) / 2), gy - 0.06, { color: MID });
	tag("C3", gx(R.C3) + 0.1, 0.06);
	tag("C2", gx(R.C2), 0.06);
	dot(s, gx(six), gy, 0.055, { fill: YEL, line: INK, lw: 1 });
	tx(s, "mean of the six models " + f3(six), gx(six) - 0.04, gy - 0.5, 2.2, 0.18, { fontFace: HEAD, fontSize: 10.5, wrap: false });
	ln(s, gx(six), gy - 0.31, gx(six), gy - 0.07, { color: INK });
	// bottom right: checks
	const cx = 5.3;
	mono(s, "CHECKS BEFORE TRUSTING IT", cx, 3.94, 4.2, { charSpacing: 0.6 });
	const checks = [
		["0.996", "correlation with the published notebook, 16 studies"],
		["0.955", "private score of the unchanged recipe; published 0.954"],
		["0.958", "with fused labels; the same for 2-, 3-, 4-model means"],
	];
	checks.forEach((c, i) => {
		const y = 4.2 + i * 0.27;
		tx(s, c[0], cx, y, 0.6, 0.25, { fontFace: HEAD, fontSize: 12.5, valign: "middle", wrap: false });
		tx(s, c[1], cx + 0.68, y, 3.55, 0.25, { fontSize: 10.5, valign: "middle", wrap: false });
	});
	s.addNotes(
		"With image-read labels our own model reached 0.927 to 0.928 on gold-58, roughly where the first-placed team's write-up puts its model before region-of-interest pooling, cropping and auxiliary heads. Rebuilding those parts ourselves would have taken longer than the time left, so we built on their published open-source design. We say this plainly, here and on the demo site: the three networks, their training script and recipe are the winning team's (Team Kangaroo, Apache-2.0 code, CC BY 4.0 labels; the 7th-place labels are CC BY-NC-SA 4.0). Our work is the label choice, the training plan, the inference pipeline, the ensemble and the evaluation. The side classifier and the detector run with the published weights; the box-pooling classifier is the part we trained, six times.\n" +
		"The pipeline, three networks in sequence. Side classifier: ResNet-18d over the coronal and axial slices, mean probability that the knee is a right knee; left knees are mirrored so all studies look alike. ROI detector: ConvNeXt-T on every slice at 160 px, features of its four stages fused top-down in three dimensions (feature-pyramid style), conditioned through FiLM layers on plane, side, fat-suppression flag and field of view; it outputs one 3D box per finding per series, averaged with a mirrored pass. It was trained on boxes that the winning team's LLM agents drew in 748 studies. Box-pooling classifier: every slice resampled to a 140 mm field on a 384-px grid (0.365 mm per pixel), a 112 mm window read at 256 px; ConvNeXt-T turns each slice into an 8x8 grid of 768-d cells; for each finding the cells are averaged with weight = (share of the cell inside the finding's box) x exp(learned in-box attention), and one linear row gives the logit. Two auxiliary heads are used in training only: one predicts which cells are covered by lesion boxes, one predicts an embedding of the report. Loss = BCE + 1.0 lesion + 0.3 report-embedding. 31.0 M parameters.\n" +
		"Fused labels: we train on the mean of the two image-read label sets. Because BCE is linear in the target, this is exactly the same as training on both sets at once with equal weight; no extra loss term.\n" +
		"Training: a builder generates the Kaggle notebook from the published script with switches for label file, fold, seed and time budget. Six runs on 2 x T4, 8 epochs, 5.4 to 6.1 hours each: A0 (first-place labels), A1 (seventh-place), A2 (fused), all fold 0; C1 to C3 = fused on folds 1 to 3. Gold-58 at the last epoch: 0.9388, 0.9389, 0.9383, 0.9369, 0.9433, 0.9397. Gold was logged every epoch but never used to pick one. Mean of six on gold-58: 0.941 [0.912, 0.965].\n" +
		"Inference: the published notebooks depended on TensorRT engines and a compiled reader tied to the authors' weights, so we wrote our own script around the three unchanged modules. Unreadable series are skipped and reported, never zero-filled; a failed batch is repeated one series at a time. Rehearsed on a free CPU session with every batch forced to fail once: 15 of 15 series recovered.\n" +
		"Checks: correlation 0.996 with the published notebook's predictions on 16 gold studies; A0 late-submitted scored 0.955 private against the published 0.954, so the pipeline reproduces the source. A1 0.957; A2 (fused) 0.958 private / 0.968 public; two-, three- and four-model means also 0.958. Differences of 0.002 are inside leaderboard noise, so read 0.955 to 0.958 as one level. Competition best 0.943, winner 0.960.\n" +
		"Honest gap: before the last two folds were trained we fixed in writing that the equal-weight mean of all six models is the final ensemble. Its notebook stopped with an error on Kaggle and has not been scored yet."
	);
}

// ================================================================== 12  synthesis
pres.addSection({ title: "Synthesis" });
{
	const s = content("Synthesis", "Fine-tuning, image-read labels and anatomy-guided pooling moved the score; most other changes did not", "SYNTHESIS  ·  WHAT MOVED THE SCORE", "all", "REPORT §3.3.3–§3.3.5, FIG. 3.5, TABLE 3.9  ·  §4.1–§4.2");
	// left: slopegraph per finding
	mono(s, "GOLD-58 AUC PER FINDING, THREE MODEL GENERATIONS", 0.5, 1.5, 4.8, { charSpacing: 0.6 });
	const gens = ["E0 frozen EfficientNet-B0", "2.5D MIL ensemble (F1-F3)", "Box-pooling, six-model mean"];
	const gl = [["Frozen", "baseline"], ["Our 2.5D", "model"], ["Box pooling,", "fused labels"]];
	const cols = [0.95, 2.3, 3.65], T = 2.52, Bm = 4.95;
	const sy = scale(0.58, 1.0, Bm, T);
	cols.forEach((x, i) => {
		tx(s, [br(gl[i][0], { fontFace: HEAD, fontSize: 10.5 }), run(gl[i][1], { fontFace: HEAD, fontSize: 10.5 })], x - 0.7, 1.76, 1.4, 0.36, { align: "center" });
		ln(s, x, T - 0.1, x, Bm, { color: HAIR });
	});
	for (const t of [0.6, 0.7, 0.8, 0.9, 1.0]) lab(s, t.toFixed(1), 0.3, sy(t), 0.4, { fontFace: MONO, fontSize: 10, color: GREY, align: "right" });
	const findings = perFinding.filter((r) => r.model === gens[0] && r.finding !== "macro").map((r) => r.finding);
	const val = (g, f) => perFinding.find((r) => r.model === g && r.finding === f).v;
	const special = { Synovitis: "hi", "Lateral OA": "red" };
	// ordinary lines first, highlighted on top
	const drawF = (f) => {
		const kind = special[f];
		const pts = gens.map((g, i) => [cols[i], sy(val(g, f))]);
		poly(s, pts, { color: kind === "red" ? RED : kind === "hi" ? INK : MID, width: kind ? 2 : 0.9 });
		pts.forEach((p) => dot(s, p[0], p[1], kind ? 0.045 : 0.028, kind === "hi" ? { fill: YEL, line: INK, lw: 1 } : { fill: kind === "red" ? RED : MID }));
	};
	findings.filter((f) => !special[f]).forEach(drawF);
	findings.filter((f) => special[f]).forEach(drawF);
	// right-end labels with collision avoidance
	const ends = findings.map((f) => ({ f, v: val(gens[2], f), y: sy(val(gens[2], f)) })).sort((a, b) => a.y - b.y);
	const minGap = 0.172;
	for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < minGap) ends[i].y = ends[i - 1].y + minGap;
	const over = ends[ends.length - 1].y - (Bm + 0.02);
	if (over > 0) ends.forEach((e) => (e.y -= over));
	for (let i = ends.length - 2; i >= 0; i--) if (ends[i + 1].y - ends[i].y < minGap) ends[i].y = ends[i + 1].y - minGap;
	ends.forEach((e) => {
		const kind = special[e.f];
		tx(s, [run(e.f + " ", { fontFace: kind ? HEAD : undefined, fontSize: 10, color: kind === "red" ? RED : INK }), run(f3(e.v), { fontFace: MONO, fontSize: 10, color: kind === "red" ? RED : kind ? INK : GREY })], cols[2] + 0.26, e.y - 0.09, 1.7, 0.18, { valign: "middle", wrap: false });
	});
	const macro = gens.map((g) => perFinding.find((r) => r.model === g && r.finding === "macro").v);
	cols.forEach((x, i) => tx(s, "macro " + f3(macro[i]), x - 0.7, 2.12, 1.4, 0.18, { fontFace: MONO, fontSize: 10, color: GREY, align: "center", wrap: false }));

	// right: ledger
	const RX = 5.75, RW = 3.75;
	mono(s, "MOVED THE SCORE", RX, 1.5, RW, { color: INK, charSpacing: 0.6 });
	const moved = [
		["+0.090", "fine-tuned encoder (held-out fold)"],
		["+0.036", "image-read labels (gold-58)"],
		["0.943 → 0.958", "box pooling + fused labels (private)"],
	];
	moved.forEach((m, i) => {
		const y = 1.76 + i * 0.3;
		rect(s, RX, y, RW, 0.27, { fill: PALE, name: "Highlighted row" });
		tx(s, m[0], RX + 0.08, y, 1.25, 0.27, { fontFace: HEAD, fontSize: 12, valign: "middle", wrap: false });
		tx(s, m[1], RX + 1.33, y, RW - 1.35, 0.27, { fontSize: 10, valign: "middle", wrap: false });
	});
	mono(s, "TESTED, MEASURED, NOT ADOPTED", RX, 2.78, RW, { color: INK, charSpacing: 0.6 });
	const not = [
		["+0.003", "higher resolution, at 1.7× training time"],
		["−0.003", "six series instead of three"],
		["+0.001", "twelve epochs instead of eight"],
		["+0.002", "transformer over tokens, 1.17× cost"],
		["≤ +0.001", "attention sharpening at inference"],
		["≤ +0.001", "per-series-type weights"],
		["−0.001", "adding our own model to the ensemble"],
		["0.958", "one model, and 2-, 3-, 4-model means"],
	];
	not.forEach((m, i) => {
		const y = 3.02 + i * 0.228;
		ln(s, RX, y, RX + RW, y, { color: HAIR });
		tx(s, m[0], RX + 0.08, y, 0.95, 0.228, { fontFace: MONO, fontSize: 10, valign: "middle", wrap: false });
		tx(s, m[1], RX + 1.0, y, RW - 1.0, 0.228, { fontSize: 10, valign: "middle", wrap: false });
	});
	ln(s, RX, 3.02 + not.length * 0.228, RX + RW, 3.02 + not.length * 0.228, { color: HAIR });
	s.addNotes(
		"Left: per-finding AUC on the 58 gold studies for three generations: the frozen baseline re-run under the Stage 3 protocol (macro 0.753), our fine-tuned 2.5D ensemble (0.906), and the six-model box-pooling mean (0.941, 95% CI 0.912 to 0.965, reported as a description; that ensemble is not yet scored on the leaderboard).\n" +
		"The synovitis line is the label story in one finding: 0.673, then 0.680, then 0.877. Our Stage 3 model had learned what reports say about synovitis, not what images show; the reports leave out about half of the synovitis that gold readers mark. Image-read labels fixed it.\n" +
		"Not everything improved: lateral OA went from 0.896 down to 0.826, and bone contusion from 0.961 to 0.928; lateral meniscal tears are weak in both models. These are open problems. With 11 to 23 positives per finding in gold-58, single-finding numbers are noisy.\n" +
		"Right: the ledger. Three things moved the score: fine-tuning the encoder (+0.090 on the held-out fold; 0.755 to 0.916 private), image-read labels on the same model (+0.036 on gold-58, CI +0.018 to +0.055), and the published box-pooling design trained on fused labels (0.943 to 0.958 private; winner 0.960). Direct evidence for anatomy-guided pooling: with the same labels and the same held-out fold, our own model scored 0.957 against 0.968 to 0.971 for the box-pooling models.\n" +
		"What did not: higher resolution (+0.003 on the fold, nothing on gold, 1.7x training time), six series, twelve epochs, a transformer head; attention sharpening (best s = 1.25, at most +0.0007; s = 3 loses 0.004 to 0.006; two sharpened versions sent to the leaderboard scored 0.956 and 0.957 against 0.957), per-series-type weights (+0.0004 to +0.0007), adding our own model to the final ensemble (-0.0012), and larger ensembles (0.958 for one, two, three and four models). Also rejected: centring scores within scanner groups (0.926 to 0.917), and in Stage 4 six blend upgrades and mirror averaging.\n" +
		"What we would say we learned: with weak labels, measure the labels before tuning the model; write the decision rule before the result; and distrust small gains on 58 studies. The one time we broke that rule, on the last day, the in-sample gain did not transfer.\n" +
		"Limits: a research prototype, trained on labels that are themselves estimates, not validated on other hospitals, not a medical device. Next steps: score the pre-registered six-model ensemble; produce image-read labels locally with a vision-language model; MOAKS grades from the OAI for the osteoarthritis findings; external validation."
	);
}

// ------------------------------------------------------------------ write
// pptxgenjs writes Office's stock palette into the theme; the deck's own colours have to be written afterwards.
async function writeThemeColors(file) {
	const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
	const zip = await JSZip.loadAsync(fs.readFileSync(file));
	const part = "ppt/theme/theme1.xml";
	const slots = ["dk1", "lt1", "dk2", "lt2", "accent1", "accent2", "accent3", "accent4", "accent5", "accent6", "hlink", "folHlink"];
	const scheme = `<a:clrScheme name="${THEME.name}">` + slots.map((k) => `<a:${k}><a:srgbClr val="${THEME.colors[k]}"/></a:${k}>`).join("") + "</a:clrScheme>";
	const xml = (await zip.file(part).async("string")).replace(/<a:clrScheme\b[\s\S]*?<\/a:clrScheme>/, () => scheme);
	if (!xml.includes(scheme)) throw new Error("theme colour scheme not found in " + part);
	// repack file entries only, in their original order (folder entries upset some cloud viewers)
	const out = new JSZip();
	for (const [name, entry] of Object.entries(zip.files)) {
		if (entry.dir) continue;
		out.file(name, name === part ? xml : await entry.async("nodebuffer"), { createFolders: false });
	}
	fs.writeFileSync(file, await out.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

(async () => {
	await pres.writeFile({ fileName: OUT });
	const skill = process.env.PPTX_SKILL_DIR;
	if (skill) {
		const { applyTheme } = require(path.join(skill, "scripts", "apply_theme.js"));
		await applyTheme(OUT, THEME);
	} else await writeThemeColors(OUT);
	console.log("wrote", OUT);
})();
