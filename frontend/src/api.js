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

// `graph` true also uses the knowledge graph; null leaves it to the server's GRAPH_SEARCH setting.
export const search = (query, types, graph = null) =>
  request("/search", json("POST", { query, limit: 10, types: types.length ? types : null, graph }));
export const ask = (query, types, graph = null) =>
  request("/ask", json("POST", { query, types: types.length ? types : null, graph }));
export const getSuggestions = () => request("/suggestions");
export const getPassage = (documentId, parentId) => request(`/documents/${documentId}/passages/${parentId}`);
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

export const getStats = () => request("/stats");
export const getEvalResults = () => request("/eval/results");
export const getEvalQuestions = () => request("/eval/questions");
export const runEval = (retrievers, k, split) => request("/eval/run", json("POST", { retrievers, k, split }));
export const getEvalJob = (id) => request(`/eval/jobs/${id}`);

// Poll an evaluation job until it finishes or fails, reporting progress on each tick.
export async function watchEval(jobId, onUpdate) {
  for (;;) {
    const job = await getEvalJob(jobId);
    onUpdate(job);
    if (job.status !== "running") return job;
    await new Promise((r) => setTimeout(r, 1000));
  }
}
export const saveEvalQuestions = (questions) => request("/eval/questions", json("PUT", { questions }));
export const checkEvalQuestions = (questions) => request("/eval/questions/check", json("POST", { questions }));
export const getGraphStats = () => request("/graph/stats");
export const rebuildGraph = () => request("/graph/rebuild", { method: "POST" });
export const findEntities = (q) => request(`/graph/entities?q=${encodeURIComponent(q)}`);
export const getEntity = (id) => request(`/graph/entity?id=${encodeURIComponent(id)}`);

export const getQuality = () => request("/quality");
export const runArtifactCheck = (sample, kinds = ["qa_pair", "factoid"]) => request("/quality/artifacts/run", json("POST", { sample, kinds }));
export const runAnswerEval = (split = "test") => request("/quality/answers/run", json("POST", { split }));
export const reviewArtifact = (id, correct) => request(`/quality/artifacts/${id}/review`, json("PUT", { correct }));
export async function watchQualityJob(jobId, onUpdate) {
  for (;;) {
    const job = await request(`/quality/jobs/${jobId}`);
    onUpdate(job);
    if (job.status !== "running") return job;
    await new Promise((r) => setTimeout(r, 1500));
  }
}
