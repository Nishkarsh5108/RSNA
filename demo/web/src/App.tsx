import { lazy, Suspense, useEffect } from "react";
import { useRoute } from "./lib/route";

// Two independent bundles: the site (WebGL hero, scroll figures) and the workstation (3D viewer, inference client).
const Landing = lazy(() => import("./landing/Landing"));
const Workstation = lazy(() => import("./workstation/Workstation"));

export default function App() {
  const route = useRoute();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [route]);
  return (
    <Suspense fallback={<div style={{ height: "100vh", background: "#000" }} />}>
      {route.startsWith("/workstation") ? <Workstation /> : <Landing />}
    </Suspense>
  );
}
