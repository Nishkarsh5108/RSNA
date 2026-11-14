import { useEffect, useState } from "react";
import { api, type Status } from "../lib/api";

export type ServerState = { kind: "offline" } | { kind: "loading"; status: Status } | { kind: "ready"; status: Status };

/** Polls the local server until the models are loaded (every 4 s), then every 20 s. */
export function useServerStatus(): ServerState | null {
  const [state, setState] = useState<ServerState | null>(null);
  useEffect(() => {
    let alive = true;
    let timer = 0;
    const tick = async () => {
      let next: ServerState;
      try {
        const s = await api.status();
        next = s.ready ? { kind: "ready", status: s } : { kind: "loading", status: s };
      } catch {
        next = { kind: "offline" };
      }
      if (!alive) return;
      setState(next);
      timer = window.setTimeout(tick, next.kind === "ready" ? 20000 : 4000);
    };
    tick();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, []);
  return state;
}

export function gpuShort(name?: string) {
  return (name ?? "").replace(/^NVIDIA\s+/i, "").replace(/^GeForce\s+/i, "").replace(/\s+GPU$/i, "");
}

export function modelCount(state: ServerState | null): number | null {
  if (!state || state.kind === "offline") return null;
  return state.status.models?.classifiers.length ?? null;
}
