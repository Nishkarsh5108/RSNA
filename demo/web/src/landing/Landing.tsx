import { useCallback, useEffect, useRef } from "react";
import Lenis from "lenis";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import "../styles/landing.css";
import Nav from "./Nav";
import Hero from "./Hero";
import Method from "./Method";
import { Contributions, Findings, Footer, Results } from "./Sections";
import { modelCount, useServerStatus } from "./useStatus";

gsap.registerPlugin(ScrollTrigger);

export default function Landing() {
  const state = useServerStatus();
  const lenis = useRef<Lenis | null>(null);

  useEffect(() => {
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced) return;
    const l = new Lenis({ lerp: 0.11, wheelMultiplier: 0.95 });
    lenis.current = l;
    l.on("scroll", ScrollTrigger.update);
    const tick = (time: number) => l.raf(time * 1000);
    gsap.ticker.add(tick);
    gsap.ticker.lagSmoothing(0);
    return () => {
      gsap.ticker.remove(tick);
      l.destroy();
      lenis.current = null;
    };
  }, []);

  // Section reveals: hairline draws across, heading slides up from a mask. Only on headers, not on every element.
  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.utils.toArray<HTMLElement>(".sec-head").forEach((el) => {
        gsap.from(el.querySelectorAll(".sec-title, .sec-lede, .sec-idx"), {
          yPercent: 30,
          opacity: 0,
          duration: 1.1,
          ease: "expo.out",
          stagger: 0.07,
          scrollTrigger: { trigger: el, start: "top 82%", once: true },
        });
      });
    });
    const refresh = window.setTimeout(() => ScrollTrigger.refresh(), 400);
    return () => {
      clearTimeout(refresh);
      ctx.revert();
    };
  }, []);

  const jump = useCallback((id: string) => {
    const el = id === "top" ? 0 : document.getElementById(id);
    if (el === null) return;
    if (lenis.current) lenis.current.scrollTo(el as HTMLElement | number, { offset: -56, duration: 1.4 });
    else if (el === 0) window.scrollTo({ top: 0 });
    else (el as HTMLElement).scrollIntoView();
  }, []);

  const models = modelCount(state);
  return (
    <div className="landing">
      <Nav state={state} onJump={jump} />
      <main>
        <Hero state={state} />
        <Findings />
        <Method models={models ?? 4} />
        <Contributions models={models} />
        <Results />
      </main>
      <Footer />
    </div>
  );
}
