// Headless Chrome over the DevTools protocol: open the local workstation, load sample study A, run inference,
// capture screenshots at 2x and write the bounding boxes of every <canvas> (the MRI views) for masking.
import { spawn } from "node:child_process";
import { writeFileSync, mkdirSync } from "node:fs";

const OUT = process.argv[2];
const PROFILE = process.argv[3];
mkdirSync(OUT, { recursive: true });
const chrome = spawn("C:/Program Files/Google/Chrome/Application/chrome.exe", [
  "--headless=new", "--remote-debugging-port=9333", `--user-data-dir=${PROFILE}`, "--no-first-run",
  "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--window-size=1600,1000", "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let targets;
for (let i = 0; i < 50; i++) {
  try { targets = await (await fetch("http://127.0.0.1:9333/json/list")).json(); if (targets.length) break; } catch {}
  await sleep(300);
}
const page = targets.find((t) => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r));
let id = 0; const pending = new Map();
ws.addEventListener("message", (ev) => { const m = JSON.parse(ev.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } });
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const evaljs = async (expr) => (await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value;
const shot = async (name) => {
  const r = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
  writeFileSync(`${OUT}/${name}.png`, Buffer.from(r.result.data, "base64"));
  const rects = await evaljs(`JSON.stringify([...document.querySelectorAll('canvas, img')].map(c => { const b = c.getBoundingClientRect(); return [b.left, b.top, b.right, b.bottom, c.tagName]; }).filter(b => b[2]-b[0] > 20 && b[3]-b[1] > 20))`);
  writeFileSync(`${OUT}/${name}.rects.json`, rects);
  console.log("shot", name, rects);
};
const clickText = (txt) => evaljs(`(() => { const all = [...document.querySelectorAll('button, a, [role=button], div, span, li')];
  const el = all.find(e => e.textContent.trim().toLowerCase() === ${JSON.stringify(txt.toLowerCase())}) ||
             all.find(e => e.textContent.trim().toLowerCase().startsWith(${JSON.stringify(txt.toLowerCase())}) && e.children.length < 4);
  if (!el) return 'missing'; (el.closest('button') || el).click(); return 'clicked'; })()`);
const waitFor = async (cond, ms) => { for (let t = 0; t < ms; t += 500) { if (await evaljs(cond)) return true; await sleep(500); } return false; };

await send("Page.enable"); await send("Runtime.enable");
await send("Emulation.setDeviceMetricsOverride", { width: 1600, height: 1000, deviceScaleFactor: 2, mobile: false });
await send("Page.navigate", { url: "http://127.0.0.1:8000/#/workstation" });
await waitFor(`document.body.innerText.includes('Public test study A')`, 20000);
await sleep(1500);
await shot("ws_intake");
console.log(await clickText("Public test study A"));
await waitFor(`[...document.querySelectorAll('button')].some(b => /run inference/i.test(b.textContent))`, 60000);
await sleep(3000);
console.log(await evaljs(`(() => { const b = [...document.querySelectorAll('button')].find(b => /run inference/i.test(b.textContent)); if (!b) return 'no button'; b.click(); return 'run'; })()`));
const done = await waitFor(`[...document.querySelectorAll('button')].some(b => /run again/i.test(b.textContent))`, 180000);
console.log("inference done", done);
await sleep(4000);
await shot("ws_result");
console.log(await clickText("Medial meniscus tear"));
await sleep(4000);
await shot("ws_finding");
console.log(await evaljs(`(() => { const els = [...document.querySelectorAll('*')].filter(e => e.scrollHeight > e.clientHeight + 40 && e.getBoundingClientRect().left > 1200);
  els.forEach(e => e.scrollTop = e.scrollHeight); return els.length; })()`));
await sleep(1500);
await shot("ws_detail");
const text = await evaljs(`document.body.innerText`);
writeFileSync(`${OUT}/ws_text.txt`, text);
ws.close(); chrome.kill();
process.exit(0);
