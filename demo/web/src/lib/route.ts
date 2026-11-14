import { useEffect, useState } from "react";

/** Hash routing: "#/" is the site, "#/workstation" the demo. */
export function useRoute(): string {
  const [route, setRoute] = useState(() => location.hash.replace(/^#/, "") || "/");
  useEffect(() => {
    const on = () => setRoute(location.hash.replace(/^#/, "") || "/");
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);
  return route;
}

export function go(route: string) {
  location.hash = route;
}
