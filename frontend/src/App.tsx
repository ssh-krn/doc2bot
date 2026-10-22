import { ChatView } from "./components/ChatView";
import { DocumentList } from "./components/DocumentList";
import { UploadButton } from "./components/UploadButton";
import { useChat } from "./hooks/useChat";
import { useDocuments } from "./hooks/useDocuments";

export default function App() {
  const documents = useDocuments();
  const chat = useChat();

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>Doc2Bot</h1>
        <UploadButton disabled={documents.uploading} onFile={documents.upload} />
        {documents.error && (
          <p className="error" role="alert">
            {documents.error}
          </p>
        )}
        <DocumentList docs={documents.docs} loading={documents.loading} onRemove={documents.remove} />
      </aside>
      <main className="main">
        <ChatView
          messages={chat.messages}
          busy={chat.busy}
          canAsk={documents.docs.length > 0}
          onSend={chat.send}
          onCancel={chat.cancel}
        />
      </main>
    </div>
  );
}
