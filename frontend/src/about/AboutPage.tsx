import { EvalResults } from "./EvalResults";
import { SupportMatrix } from "./SupportMatrix";
import { METHODS_NOTE, MODES, RERANKERS, TECHNIQUES, WORKFLOW, type Item } from "./content";

function ItemCard({ item }: { item: Item }) {
  return (
    <div className="technique">
      <h4>{item.name}</h4>
      <p>
        <span className="label">What it does.</span> {item.what}
      </p>
      <p>
        <span className="label">Why.</span> {item.why}
      </p>
      {item.limits && (
        <p>
          <span className="label">Limits.</span> {item.limits}
        </p>
      )}
      <p className="where">
        <code>{item.where}</code>
      </p>
    </div>
  );
}

export function AboutPage() {
  return (
    <article className="about">
      <header>
        <h2>About Doc2Bot</h2>
        <p className="lead">
          Doc2Bot answers questions about your own documents and cites the exact page each answer came from. It handles
          PDFs, Word, PowerPoint, Excel and images, including tables, figures and equations, and offers several RAG
          modes so you can compare how they answer.
        </p>
        <nav className="toc" aria-label="On this page">
          <a href="#workflow">Workflow</a>
          <a href="#modes">RAG modes</a>
          <a href="#techniques">Retrieval techniques</a>
          <a href="#rerankers">Rerankers</a>
          <a href="#matrix">Support and evidence matrix</a>
        </nav>
      </header>

      <section id="workflow">
        <h3>How the workflow runs</h3>
        <p>
          Every document goes through the same steps, from upload to a cited answer. The pipeline options you choose
          change the middle steps. The rest always run.
        </p>
        <ol className="workflow">
          {WORKFLOW.map((step) => (
            <li key={step.title}>
              <strong>{step.title}.</strong> {step.body}
            </li>
          ))}
        </ol>
      </section>

      <section id="modes">
        <h3>RAG modes</h3>
        <p>
          A mode decides how the system decides what to retrieve and whether to trust it. Traditional RAG is the
          baseline. The other five add a step: grading, self-checking, graph retrieval, another search round, or
          community summaries.
        </p>
        <div className="techniques">
          {MODES.map((m) => (
            <ItemCard key={m.name} item={m} />
          ))}
        </div>
      </section>

      <section id="techniques">
        <h3>Retrieval techniques</h3>
        <p>
          Techniques can be combined with any mode. Hybrid search is named after the mode it runs under, because the
          same keyword and vector search works differently in each one.
        </p>
        <div className="techniques">
          {TECHNIQUES.map((t) => (
            <ItemCard key={t.name} item={t} />
          ))}
        </div>
      </section>

      <section id="rerankers">
        <h3>Rerankers</h3>
        <p>{METHODS_NOTE}</p>
        <div className="techniques">
          {RERANKERS.map((r) => (
            <ItemCard key={r.name} item={r} />
          ))}
        </div>
      </section>

      <section id="matrix">
        <h3>Support and evidence matrix</h3>
        <p>
          The matrix sets what each mode, technique and reranker is claimed to do against independent checks, then shows
          how the two models in this build fit each one. Evidence strength is my reading of the sources, not a score.
        </p>
        <SupportMatrix />
      </section>

      <section id="evaluation">
        <EvalResults />
      </section>
    </article>
  );
}
