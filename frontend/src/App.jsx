import { useEffect, useState } from "react";
import { NavLink, Route, Routes, useLocation } from "react-router-dom";
import { getBranding, health } from "./api.js";
import { Logo } from "./components.jsx";
import Ask from "./pages/Ask.jsx";
import Ingest from "./pages/Ingest.jsx";
import AddDocument from "./pages/AddDocument.jsx";
import ApiDocs from "./pages/ApiDocs.jsx";
import Passage from "./pages/Passage.jsx";

const Icon = ({ d }) => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={d} /></svg>
);

const PAGES = [
  { to: "/", label: "Ask", icon: "M21 21l-4.3-4.3M11 18a7 7 0 100-14 7 7 0 000 14z" },
  { to: "/ingest", label: "Ingest", icon: "M12 3v12m0 0l-4-4m4 4l4-4M4 17v3h16v-3" },
  { to: "/add", label: "Add documents", icon: "M14 3H6a2 2 0 00-2 2v14a2 2 0 002 2h12a2 2 0 002-2V9zM14 3v6h6M12 12v6M9 15h6" },
  { to: "/api-docs", label: "API docs", icon: "M16 18l6-6-6-6M8 6l-6 6 6 6" },
];

export default function App() {
  const [branding, setBranding] = useState({ name: "Support Search", tagline: "", logo_url: "" });
  const [open, setOpen] = useState(false);
  const [online, setOnline] = useState(null);
  const { pathname } = useLocation();

  useEffect(() => setOpen(false), [pathname]);
  useEffect(() => {
    getBranding().then((b) => { setBranding(b); document.title = b.name; }).catch(() => {});
  }, []);
  useEffect(() => {
    const check = () => health().then(() => setOnline(true)).catch(() => setOnline(false));
    check();
    const timer = setInterval(check, 15000);
    return () => clearInterval(timer);
  }, []);

  const title = pathname.startsWith("/passage") ? "Passage" : PAGES.find((p) => p.to === pathname)?.label ?? "";

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-20 flex h-14 items-center gap-3.5 border-b border-line bg-panel px-4 md:px-5">
        <button className="rounded-lg border border-line p-1.5 md:hidden" aria-label="Toggle menu" aria-expanded={open} onClick={() => setOpen(!open)}>
          <Icon d={open ? "M6 6l12 12M18 6L6 18" : "M4 6h16M4 12h16M4 18h16"} />
        </button>
        <span className="flex items-center gap-2.5">
          <Logo branding={branding} size="size-9" />
          <span className="leading-tight">
            <span className="block font-semibold tracking-tight">{branding.name}</span>
            {branding.tagline && <span className="block text-xs text-muted">{branding.tagline}</span>}
          </span>
        </span>
        <span className="hidden border-l border-line pl-3.5 text-muted md:inline">{title}</span>
        <span className="ml-auto inline-flex items-center gap-2 rounded-full border border-line px-2.5 py-1 text-xs text-muted" title="API status">
          <i className={`size-2 rounded-full ${online === null ? "bg-muted" : online ? "bg-ok" : "bg-bad"}`} />
          <span className="hidden sm:inline">{online === null ? "Checking" : online ? "API online" : "API offline"}</span>
        </span>
      </header>

      <div className="flex flex-1">
        {open && <div className="fixed inset-0 top-14 z-20 bg-black/45 md:hidden" onClick={() => setOpen(false)} />}
        <nav
          aria-label="Main"
          className={`fixed bottom-0 left-0 top-14 z-30 flex w-58 flex-col gap-1 bg-nav p-3 transition-transform md:sticky md:h-[calc(100vh-3.5rem)] md:translate-x-0 md:self-start ${open ? "" : "-translate-x-full"}`}
        >
          {PAGES.map((p) => (
            <NavLink
              key={p.to}
              to={p.to}
              end={p.to === "/"}
              className={({ isActive }) => `flex items-center gap-3 rounded-lg px-3 py-2 ${isActive ? "bg-accent text-white" : "text-slate-400 hover:bg-white/10 hover:text-white"}`}
            >
              <Icon d={p.icon} />{p.label}
            </NavLink>
          ))}
        </nav>
        <main className="min-w-0 flex-1 px-3.5 py-5 md:px-7 md:py-7">
          <div className="mx-auto max-w-4xl">
            <Routes>
              <Route path="/" element={<Ask />} />
              <Route path="/ingest" element={<Ingest />} />
              <Route path="/add" element={<AddDocument />} />
              <Route path="/api-docs" element={<ApiDocs />} />
              <Route path="/passage/:documentId/:parentId" element={<Passage />} />
            </Routes>
          </div>
        </main>
      </div>

      <footer className="flex flex-wrap items-center justify-between gap-x-6 gap-y-1 border-t border-line bg-panel px-3.5 py-3 text-xs text-muted md:px-7">
        <span>Developed by Albert Pinto for the UC Berkeley Capstone project</span>
        <span>© {new Date().getFullYear()} {branding.name}</span>
      </footer>
    </div>
  );
}
