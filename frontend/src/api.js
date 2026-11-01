async function request(path, options) {
  const response = await fetch(`/api${path}`, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body;
}

const json = (method, body) => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const search = (query, types) =>
  request("/search", json("POST", { query, limit: 10, types: types.length ? types : null }));
export const ask = (query, types) =>
  request("/ask", json("POST", { query, types: types.length ? types : null }));
export const ingestFiles = (files) => {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  return request("/ingest/files", { method: "POST", body: form });
};
export const ingestFolder = () => request("/ingest/folder", { method: "POST" });
export const ingestText = (title, text) => request("/ingest/text", json("POST", { title, text }));
export const health = () => request("/health");
export const getJob = (id) => request(`/jobs/${id}`);
export const listDocuments = () => request("/documents");
export const deleteDocument = (id) => request(`/documents/${id}`, { method: "DELETE" });

// Poll a job until it finishes, reporting progress on each tick.
export async function watchJob(jobId, onUpdate) {
  for (;;) {
    const job = await getJob(jobId);
    onUpdate(job);
    if (job.status === "done") return job;
    await new Promise((r) => setTimeout(r, 1500));
  }
}
export const getBranding = () => request("/branding");
export const getOpenApi = () => request("/openapi.json");
