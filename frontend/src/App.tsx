import { useState } from "react";
import { AboutPage } from "./about/AboutPage";
import { ChatView } from "./components/ChatView";
import { DocumentList } from "./components/DocumentList";
import { UploadButton } from "./components/UploadButton";
import { useChat } from "./hooks/useChat";
import { useDocuments } from "./hooks/useDocuments";
import { useOptions } from "./hooks/useOptions";
import { DEFAULT_SETTINGS, type PipelineSettings } from "./types";

type View = "chat" | "about";

export default function App() {
  const [view, setView] = useState<View>("chat");
  const [settings, setSettings] = useState<PipelineSettings>(DEFAULT_SETTINGS);
  const documents = useDocuments();
  const chat = useChat();
  const options = useOptions();

  return (
    <div className="shell">
      <nav className="topbar" aria-label="Main">
        <span className="brand">Doc2Bot</span>
        <div className="tabs" role="tablist">
          {(["chat", "about"] as const).map((v) => (
            <button
              key={v}
              type="button"
              role="tab"
              aria-selected={view === v}
              className={`tab${view === v ? " active" : ""}`}
              onClick={() => setView(v)}
            >
              {v === "chat" ? "Chat" : "About"}
            </button>
          ))}
        </div>
      </nav>

      {view === "chat" ? (
        <div className="app">
          <aside className="sidebar">
            <UploadButton disabled={documents.uploading} onFile={documents.upload} />
            {documents.error && (
              <p className="error" role="alert">
                {documents.error}
              </p>
            )}
            {options.error && (
              <p className="error" role="alert">
                {options.error}
              </p>
            )}
            <DocumentList docs={documents.docs} loading={documents.loading} onRemove={documents.remove} />
          </aside>
          <main className="main">
            <ChatView
              messages={chat.messages}
              busy={chat.busy}
              canAsk={documents.docs.length > 0}
              catalog={options.catalog}
              settings={settings}
              onSettings={setSettings}
              onSend={(q) => chat.send(q, settings)}
              onCancel={chat.cancel}
            />
          </main>
        </div>
      ) : (
        <main className="about-wrap">
          <AboutPage />
        </main>
      )}
    </div>
  );
}
