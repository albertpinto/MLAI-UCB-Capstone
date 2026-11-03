import { Link } from "react-router-dom";
import { Download } from "lucide-react";

// Screenshots live in frontend/public/guide/ (made by scripts/screenshots.sh from the running app).
function Shot({ file, alt, children }) {
  return (
    <figure className="my-4 overflow-hidden rounded-2xl border border-line bg-panel shadow-card">
      <a href={`/guide/${file}`} target="_blank" rel="noreferrer"><img src={`/guide/${file}`} alt={alt} loading="lazy" className="w-full" /></a>
      <figcaption className="border-t border-line px-4 py-2 text-sm text-muted">{children ?? alt}</figcaption>
    </figure>
  );
}

function Steps({ children }) {
  return <ol className="my-3 list-decimal space-y-1.5 pl-6">{children}</ol>;
}

const Tip = ({ children }) => <p className="my-3 rounded-xl border border-line bg-soft px-4 py-3 text-sm">{children}</p>;

const SECTIONS = [
  ["start", "Getting started"],
  ["ask", "Ask"],
  ["documents", "Documents"],
  ["dashboard", "Dashboard"],
  ["questions", "Questions"],
  ["graph", "Knowledge graph"],
  ["quality", "Quality checks"],
  ["operate", "Running and configuring"],
  ["help", "Troubleshooting"],
];

export default function Guide() {
  return (
    <div className="lg:flex lg:gap-8">
      <nav aria-label="On this page" className="mb-6 lg:sticky lg:top-24 lg:mb-0 lg:w-48 lg:shrink-0 lg:self-start">
        <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.08em] text-muted">On this page</p>
        <ul className="flex flex-wrap gap-x-4 gap-y-1 text-sm lg:block lg:space-y-1">
          {SECTIONS.map(([id, label]) => <li key={id}><a className="text-muted hover:text-accent" href={`#${id}`}>{label}</a></li>)}
        </ul>
      </nav>

      <article className="min-w-0 max-w-3xl flex-1 space-y-10 leading-relaxed [&_code]:rounded [&_code]:bg-soft [&_code]:px-1 [&_h2]:scroll-mt-24 [&_h2]:text-xl [&_h2]:font-semibold [&_h3]:mt-5 [&_h3]:font-semibold [&_ul]:my-3 [&_ul]:list-disc [&_ul]:space-y-1.5 [&_ul]:pl-6">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">User guide</h1>
          <p className="hint mt-1">How to load documents, ask questions in everyday words, and check how well the system works. Click any screenshot to open it full size.</p>
          <a className="btn no-print mt-3 inline-flex items-center gap-2 !py-2" href="/user-guide.pdf" download="User guide.pdf"><Download size={16} />Download as PDF</a>
        </div>

        <section id="start">
          <h2>Getting started</h2>
          <p>
            This app answers questions from your own documents. You ask “Who appoints the Governor?” in plain language; it finds the matching passages even when the
            document uses different words, then writes a short answer with numbered citations. If the documents do not contain the answer, it says so instead of guessing.
          </p>
          <h3>The six steps</h3>
          <Steps>
            <li><b>Start the app</b> with <code>./start.sh</code> (Windows: <code>start.ps1</code>). It opens the API on port 8000 and this page on port 5173.</li>
            <li><b>Add documents</b> on the <Link className="text-accent" to="/documents">Documents</Link> page (PDF, Word or text).</li>
            <li><b>Ask</b> a question on the <Link className="text-accent" to="/">Ask</Link> page.</li>
            <li><b>Check the answer</b> by opening the cited passage.</li>
            <li><b>Measure</b> retrieval on the <Link className="text-accent" to="/dashboard">Dashboard</Link>, using the labelled <Link className="text-accent" to="/questions">Questions</Link>.</li>
            <li><b>Check the LLM’s work</b> on <Link className="text-accent" to="/quality">Quality checks</Link>.</li>
          </Steps>
          <p>The coloured dot at the top right shows whether the API is online. If it says “API offline”, start the app again.</p>
          <h3>What happens to a document</h3>
          <ul>
            <li>It is cut into <b>sections</b> (about half a page, following the document’s headings) and each section into small <b>chunks</b> of about 65 words.</li>
            <li>An LLM rewrites vague words in each chunk (“It” becomes “The President”) and writes <b>Q&A pairs</b>, <b>factoids</b> and <b>relations</b> (who does what to whom).</li>
            <li>Everything is turned into vectors and stored. A search matches a small chunk, then the app reads its whole section, so answers have context.</li>
          </ul>
        </section>

        <section id="ask">
          <h2>Ask</h2>
          <Shot file="ask.png" alt="The Ask page with a question, a written answer with citations, and source cards" />
          <Steps>
            <li>Type a question in everyday words, or click one of the suggested questions. Press Enter.</li>
            <li>The <b>answer</b> appears on the left. Numbered chips like <b>1</b> are citations: select one to highlight its source card.</li>
            <li>The <b>Sources</b> panel on the right has a card for each passage that helped, rewritten as a short readable answer, with its document, section and page. A bar shows how close the passage is to your question. Cards marked <b>used</b> were cited in the answer; other helpful passages can appear without the mark.</li>
            <li><b>View passage</b> opens the full original text of the section, with buttons to read the previous and next one.</li>
          </Steps>
          <h3>Filters</h3>
          <ul>
            <li><b>Search in</b> chips limit what is searched: Q&A, Passages, Facts, Relations, Summaries, Topics, Graph summaries. “Everything” (the default) searches the usual types (relations and graph summaries only when you pick them).</li>
            <li><b>Knowledge graph</b> adds passages tied to the entities named in your question. On our test set it neither helped nor hurt, so it starts off. Passages it contributed are tagged “via graph”.</li>
          </ul>
          <Tip>If the answer reads “I couldn’t find this in the documentation”, nothing relevant was retrieved. That is deliberate: the system abstains rather than invent an answer. Try different words, or add the missing document.</Tip>
          <p>The question is kept in the page address (<code>/?q=…</code>), so you can bookmark or share an answer.</p>
        </section>

        <section id="documents">
          <h2>Documents</h2>
          <Shot file="documents.png" alt="The Documents page with the upload box and the three tabs" />
          <h3>Add documents</h3>
          <ul>
            <li><b>Upload</b>: drag files onto the box or click it. PDF, DOCX and TXT work. You can choose many files at once.</li>
            <li><b>Ingest the docs folder</b> reads every file in the server’s <code>DOCS_DIR</code> (<code>data/docs</code> by default).</li>
            <li><b>Paste text</b>: give a title and paste the text. The title becomes the citation shown with answers.</li>
          </ul>
          <p>Ingestion runs in the background and updates every second, so you can watch it work. For each file the table shows what it is doing now (reading the file, splitting, enriching chunks with the LLM, writing the summary and glossary, storing) and the counts grow as it goes: chunks, Q&amp;A pairs, facts, triplets and terms. While chunks are being enriched the line under the progress bar says how many are done, for example “120 of 2358 chunks”. Reading a large PDF shows no counts for a few minutes; that is the slow first step, not a hang.</p>
          <p>How long the enrichment takes depends on the LLM provider you chose (see <a className="text-accent" href="#operate">Running and configuring</a>). Finished work is cached, so a repeated or resumed run is fast.</p>
          <Tip>The same text is never stored twice (it is recognised by a hash and shown as “skipped”).</Tip>
          <h3>Library, versions and expiry</h3>
          <Shot file="library.png" alt="The Library tab listing documents with version and expiry" />
          <ul>
            <li>The <b>Library</b> tab lists every document with its <b>version</b>, when it was stored and when it expires.</li>
            <li><b>Updating</b>: add a file with the same name or title but changed text. The new version is built first, then the old one is deleted, so a failed update loses nothing. The version number goes up by one.</li>
            <li><b>Expiry</b>: set <code>DOCUMENT_TTL_DAYS</code> and documents stored from then on expire after that many days. Expired documents are flushed every <code>FLUSH_INTERVAL_HOURS</code> (24 by default) or on demand with <code>POST /api/documents/flush</code>. With the default 0, nothing expires.</li>
            <li><b>Delete</b> (the bin icon) removes a document and everything made from it. This cannot be undone.</li>
          </ul>
        </section>

        <section id="dashboard">
          <h2>Dashboard</h2>
          <Shot file="dashboard.png" alt="The Dashboard comparing retrievers by Recall, MRR and nDCG" />
          <p>The Dashboard answers “which way of searching finds the right passage most often?” on the labelled questions.</p>
          <h3>Run an evaluation</h3>
          <Steps>
            <li>Tick the retrievers, choose <b>k</b>, and choose the questions: the <b>test split</b> (held-out Parts, the honest number), train, or all.</li>
            <li>Click <b>Run evaluation</b>. The first run embeds the whole index and takes a few minutes; later runs are quick. Progress is shown.</li>
            <li>Read the comparison table and charts. Download the per-question table as CSV.</li>
          </Steps>
          <h3>The retrievers</h3>
          <ul>
            <li><b>BM25</b>: keyword matching. The baseline.</li>
            <li><b>Base MiniLM</b>: the stock embedding model. <b>Fine-tuned MiniLM</b>: the same model trained on this corpus (appears when configured).</li>
            <li><b>Hybrid</b>: BM25 and the base model combined by reciprocal rank fusion.</li>
            <li><b>+ cross-encoder rerank</b>: a second model re-orders the top 30 results. Ask uses base + rerank.</li>
            <li><b>+ knowledge graph</b>: adds graph candidates before ranking.</li>
          </ul>
          <h3>The metrics</h3>
          <ul>
            <li><b>Recall@k</b>: share of questions where the right section is in the top k.</li>
            <li><b>MRR</b>: 1 divided by the rank of the first right section, averaged. Higher means the right answer sits nearer the top.</li>
            <li><b>nDCG@k</b>: like MRR but rewards every right section and discounts lower ranks.</li>
          </ul>
          <p>The per-question table shows the rank each retriever gave the right section; misses are highlighted. The index statistics show how many points of each type exist and how big the chunks are.</p>
          <Tip>With few test questions, a difference of one or two questions is noise. Treat small gaps as ties.</Tip>
        </section>

        <section id="questions">
          <h2>Questions</h2>
          <Shot file="questions.png" alt="The Questions page listing labelled evaluation questions" />
          <p>This is the answer key for the Dashboard. Each question lists <b>evidence</b>: a phrase copied from the passage that answers it. Any section containing the phrase counts as correct, so the labels survive re-chunking.</p>
          <Steps>
            <li>Click <b>Add</b>. Give an id, the question in everyday words, and one evidence phrase per line.</li>
            <li>Choose <b>test</b> or <b>train</b>. Train questions may be used to fine-tune the model; the Parts of the document holding a test question are kept out of training so the result is not inflated.</li>
            <li>Under each question the page shows which sections contain its evidence. <span className="text-bad">Not found in any section</span> means a typo or a changed document: an evaluation would leave it out.</li>
            <li>Click <b>Save changes</b>. <b>Import</b> and <b>Export</b> read and write the JSON file.</li>
          </Steps>
          <Tip>The most honest test questions are written by someone who has not read the documents.</Tip>
        </section>

        <section id="graph">
          <h2>Knowledge graph</h2>
          <Shot file="graph.png" alt="The Knowledge graph page showing an entity, its neighbours and its relations" />
          <p>The graph is built from the relations the LLM found in each chunk (“Governor — appoints — Chief Minister”). Different spellings of one thing merge into one entity, and every relation remembers the chunk that stated it.</p>
          <ul>
            <li>A graph with thousands of entities is unreadable as one picture, so this page shows <b>one entity at a time</b>: search for it, or click a bar in the most-connected chart.</li>
            <li>The network shows the entity in the centre and its closest neighbours around it. <b>Click a neighbour</b> to move there.</li>
            <li>The table lists the relations with the section, page and a link to the source passage. Always check the source: the relations are only as good as the LLM’s reading.</li>
            <li><b>Rebuild</b> recreates the graph from the stored relations. It also happens automatically after every upload or deletion.</li>
          </ul>
        </section>

        <section id="quality">
          <h2>Quality checks</h2>
          <Shot file="quality.png" alt="The Quality checks page with answer scores and a generated item next to its source" />
          <p>An LLM wrote the Q&A pairs, factoids and answers, so some are wrong. This page measures how many.</p>
          <h3>Answers</h3>
          <p>Click <b>Evaluate answers</b>. Every test question is answered as Ask would, and a second LLM call grades each answer against the passages it was given:</p>
          <ul>
            <li><b>Faithfulness</b>: every claim is supported by the passages.</li>
            <li><b>Relevance</b>: the answer addresses the question.</li>
            <li><b>Wrongly abstained</b>: questions that have an answer but the system said “I couldn’t find this”.</li>
            <li><b>Correctly abstained</b>: off-topic questions (listed in <code>eval/unanswerable.json</code>) that were refused, as they should be.</li>
          </ul>
          <h3>Generated Q&A pairs and factoids</h3>
          <Steps>
            <li>Choose how many random items to check and click <b>Check a sample</b>. Each costs one LLM call.</li>
            <li>For each item the page shows the generated text beside the chunk it came from, and the LLM’s verdict.</li>
            <li>Read the source and click <b>Correct</b> or <b>Incorrect</b> (click again to clear). Use “Flagged by the LLM” to look at the likely errors first.</li>
            <li>The tiles show the LLM-flagged error rate, your own error rate and how often you agree. <b>Report your own rate</b>: it is the trustworthy one, and a sample of 50 to 100 is enough to be useful.</li>
          </Steps>
        </section>

        <section id="operate">
          <h2>Running and configuring</h2>
          <h3>Start and stop</h3>
          <ul>
            <li><code>./start.sh</code> starts Qdrant (Docker), the API and this page. <code>./stop.sh</code> stops the API and page; <code>./stop.sh --all</code> also stops Qdrant. Your data is kept.</li>
            <li><b>Without Docker</b> (for example on Windows): set <code>QDRANT_PATH=data/qdrant</code> in <code>.env</code>. The index is then a folder inside the project. To copy an existing index into it, run <code>uv run python -m capstone.export_index data/qdrant</code> once. The folder can be zipped and moved to another computer.</li>
            <li>On Windows use <code>start.ps1</code> and <code>stop.ps1</code>.</li>
          </ul>
          <h3>Settings that matter</h3>
          <p>Everything is in <code>.env</code> (see <code>.env.example</code>). The ones you are most likely to change:</p>
          <ul>
            <li><code>LLM_PROVIDER</code>: which service runs the LLM steps, <code>groq</code> (default) or <code>openai</code>. This one line decides; the other provider’s lines are ignored.</li>
            <li><code>GROQ_API_KEY</code>, <code>GROQ_MODEL</code> or <code>OPENAI_API_KEY</code>, <code>OPENAI_MODEL</code>: the key (required) and model of the provider you use.</li>
            <li><code>LLM_WORKERS</code>, <code>LLM_MAX_CONCURRENT</code>: how many LLM calls run at once. Keep them equal: 4 for Groq, up to 24 for OpenAI.</li>
            <li><code>COLLECTION</code>: the name of the index. <b>Use a new name</b> whenever you change the embedding model, then ingest again.</li>
            <li><code>DOCUMENT_TTL_DAYS</code>, <code>FLUSH_INTERVAL_HOURS</code>: expiry of new documents.</li>
            <li><code>GRAPH_SEARCH</code>: use the knowledge graph in every search (off by default).</li>
            <li><code>ANSWER_MIN_SCORE</code>: how similar a passage must be to count. Raise it to abstain more often.</li>
          </ul>
          <h3>Choosing the LLM provider</h3>
          <p>The LLM writes the Q&amp;A pairs, facts and triplets for every chunk, writes the answers, and checks its own work. You can use either provider. Put the key of the one you want in <code>.env</code>, set <code>LLM_PROVIDER</code>, then restart (<code>./stop.sh</code> and <code>./start.sh</code>); the API reads <code>.env</code> only when it starts.</p>
          <ul>
            <li><b>Groq</b> (default): fast per call, but it limits how much text it writes per minute, so more than 4 parallel calls only causes retries. Use <code>LLM_WORKERS=4</code> and <code>LLM_MAX_CONCURRENT=4</code>.</li>
            <li><b>OpenAI</b>: no such cap at the account level we tested, so it speeds up with more parallel calls. Use up to <code>LLM_WORKERS=24</code> and <code>LLM_MAX_CONCURRENT=24</code>. If you see rate-limit (429) errors, lower them; what your account allows depends on its usage tier.</li>
          </ul>
          <p>Measured on 213 chunks of the Constitution, and scaled to the whole document (an estimate, not a promise): OpenAI <code>gpt-4o-mini</code> with 24 calls at once, about 7 minutes; Groq <code>qwen/qwen3.8-27b</code> at its best, about 17 minutes or more. Speed is not quality: both gave similar counts of Q&amp;A pairs, facts and triplets, but how correct they are is what the Quality checks page measures. OpenAI is a paid service; the whole Constitution costs roughly a dollar with <code>gpt-4o-mini</code> (an estimate).</p>
          <Tip>Switching to a different model means the chunks are enriched again, because the cache is kept per model. Switching back to a model you already used reuses its cached results.</Tip>
          <p>To time both on your own machine, run <code>uv run python scripts/bench_ingest.py</code> once with each provider set in <code>.env</code>. It enriches the same pages from your PDF in a throwaway index and prints the seconds per chunk. Nothing is written to your real index.</p>
          <p><code>.env</code> wins over variables already exported in your terminal. A key left in <code>~/.bashrc</code> for another project will not override the one in this project’s <code>.env</code>.</p>
        </section>

        <section id="help">
          <h2>Troubleshooting</h2>
          <ul>
            <li><b>“API offline”</b>: the API is not running or still loading its models (the first start downloads them). Wait a minute, then run <code>./start.sh</code> again.</li>
            <li><b>The key is missing</b> (“GROQ_API_KEY is not set” or “OPENAI_API_KEY is not set”): copy <code>.env.example</code> to <code>.env</code> and put the key of the provider named in <code>LLM_PROVIDER</code> in it.</li>
            <li><b>“Invalid API Key” (401)</b>: the key is wrong or was revoked. Create a new one at the provider, put it in <code>.env</code> and restart. Nothing is retried for this error, so it shows at once.</li>
            <li><b>“model does not exist or you do not have access” (404)</b>: <code>GROQ_MODEL</code> or <code>OPENAI_MODEL</code> names a model your key cannot use. Pick another one and restart. This is also shown at once, without retries.</li>
            <li><b>Ingestion is slow, or you see rate-limit (429) messages in the API log</b>: the provider limits how much it will write per minute. Run it again (cached work is not repeated), use fewer parallel calls on Groq (4), or switch to OpenAI, which handles more at once.</li>
            <li><b>The Dashboard has no graph retrievers</b>: the graph is built after the first ingest. Use Rebuild on the Knowledge graph page.</li>
            <li><b>Answers say “I couldn’t find this” for something that is in the documents</b>: use words closer to the document, or lower <code>ANSWER_MIN_SCORE</code> a little.</li>
            <li><b>A question is “Not found in any section”</b> on the Questions page: its evidence phrase is not in the current text. Fix the phrase or remove the question.</li>
            <li>Technical details for developers are on the <Link className="text-accent" to="/api-docs">API docs</Link> page.</li>
          </ul>
        </section>
      </article>
    </div>
  );
}
