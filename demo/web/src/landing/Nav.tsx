import { PROJECT } from "../lib/findings";
import { gpuShort, type ServerState } from "./useStatus";

interface Props {
  state: ServerState | null;
  onJump: (id: string) => void;
}

const LINKS = [
  ["findings", "Findings"],
  ["method", "Method"],
  ["contributions", "Contributions"],
  ["results", "Results"],
] as const;

export function StatusReadout({ state }: { state: ServerState | null }) {
  if (!state) return <span className="label">Checking local server…</span>;
  if (state.kind === "offline")
    return (
      <span className="label nav-status">
        <span className="dot" /> Server offline · start demo/server/app.py
      </span>
    );
  const s = state.status;
  if (state.kind === "loading")
    return (
      <span className="label nav-status">
        <span className="dot dot--live" /> Loading models{s.error ? ` · ${s.error}` : "…"}
      </span>
    );
  const gpu = s.gpu ? `${gpuShort(s.gpu.name)} · ${s.gpu.totalGB.toFixed(1)} GB` : s.device?.toUpperCase();
  return (
    <span className="label label--fg nav-status" title={`Local inference server at ${s.host}`}>
      <span className="dot dot--accent" /> Local <span className="ns-long">· {gpu} </span>· {s.models?.classifiers.length ?? 0} models
    </span>
  );
}

export default function Nav({ state, onJump }: Props) {
  return (
    <header className="nav">
      <a
        className="nav-mark"
        href="#/"
        onClick={(e) => {
          e.preventDefault();
          onJump("top");
        }}
        aria-label={PROJECT.name}
      >
        <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
          <rect x="0.5" y="0.5" width="17" height="17" fill="none" stroke="currentColor" />
          <rect x="6" y="6" width="6" height="6" fill="var(--accent)" />
        </svg>
        <span className="label label--fg">{PROJECT.name}</span>
      </a>
      <nav className="nav-links" aria-label="Sections">
        {LINKS.map(([id, label], i) => (
          <button key={id} className="label nav-link" onClick={() => onJump(id)}>
            <span className="nav-idx">{String(i + 1).padStart(2, "0")}</span> {label}
          </button>
        ))}
      </nav>
      <div className="nav-right">
        <StatusReadout state={state} />
        <a className="btn btn--sm btn--accent nav-cta" href="#/workstation">
          Workstation <span className="arrow">→</span>
        </a>
      </div>
    </header>
  );
}
