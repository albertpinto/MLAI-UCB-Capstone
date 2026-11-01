import { ingestFolder } from "../api.js";
import { UploadBox } from "../components.jsx";

export default function Ingest() {
  return (
    <>
      <h1 className="mb-4 text-2xl font-semibold tracking-tight">Ingest documents</h1>
      <UploadBox>
        {(run, running) => (
          <p className="my-4">
            <button className="btn" disabled={running} onClick={() => run(ingestFolder)}>Ingest the docs folder</button>
            <span className="hint ml-2">Reads every file in the server's DOCS_DIR.</span>
          </p>
        )}
      </UploadBox>
    </>
  );
}
