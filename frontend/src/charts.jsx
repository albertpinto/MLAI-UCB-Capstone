import { useEffect, useState } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-cartesian-dist-min";

const Plot = createPlotlyComponent(Plotly);
const TOKENS = { text: "--text", muted: "--muted", grid: "--line", panel: "--panel", s1: "--s1", s2: "--s2", s3: "--s3" };

// Plotly needs real colour strings, so read the CSS tokens and re-read them when the OS theme changes.
function readTheme() {
  const css = getComputedStyle(document.documentElement);
  return Object.fromEntries(Object.entries(TOKENS).map(([k, v]) => [k, css.getPropertyValue(v).trim()]));
}
export function useTheme() {
  const [theme, setTheme] = useState(readTheme);
  useEffect(() => {
    const query = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => setTheme(readTheme());
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);
  return theme;
}

// A retriever keeps its colour whatever else is selected (fixed order: BM25, base, fine-tuned).
export const RETRIEVER_COLOUR = { bm25: "s1", base: "s2", tuned: "s3" };

export function Chart({ data, layout, height = 320, label }) {
  const t = useTheme();
  const axis = { gridcolor: t.grid, linecolor: t.grid, zerolinecolor: t.grid, tickfont: { color: t.muted }, title: { font: { color: t.muted } }, fixedrange: true };
  return (
    <div role="img" aria-label={label}>
      <Plot
        data={data(t)}
        layout={{
          height, autosize: true, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
          font: { family: '"Inter Variable", system-ui, sans-serif', size: 12, color: t.text },
          margin: { l: 56, r: 16, t: 24, b: 48 },
          legend: { orientation: "h", y: -0.2, font: { color: t.muted } },
          hoverlabel: { bgcolor: t.panel, bordercolor: t.grid, font: { color: t.text } },
          ...layout(t),
          xaxis: { ...axis, ...layout(t).xaxis },
          yaxis: { ...axis, ...layout(t).yaxis },
        }}
        config={{ displaylogo: false, responsive: true, displayModeBar: "hover", modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"] }}
        style={{ width: "100%" }}
        useResizeHandler
      />
    </div>
  );
}
