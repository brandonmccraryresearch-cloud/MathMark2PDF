import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import Header from './components/Header';
import Toolbar from './components/Toolbar';
import { ViewMode, DocumentState, InsertType } from './types';
import { insertMarkdown, defaultMarkdown } from './lib/markdown-utils';

// Shared component configuration
const markdownComponents: any = {
  a: ({node, ...props}: any) => <a {...props} className="text-brand-red hover:underline decoration-brand-gold/50" target="_blank" rel="noopener noreferrer" />,
  img: ({node, ...props}: any) => <img {...props} className="rounded-lg shadow-md max-w-full" loading="lazy" />,
  table: ({node, ...props}: any) => <div className="overflow-x-auto my-4"><table {...props} className="min-w-full divide-y divide-gray-200 border" /></div>,
  thead: ({node, ...props}: any) => <thead {...props} className="bg-gray-50" />,
  th: ({node, ...props}: any) => <th {...props} className="px-6 py-3 text-left text-xs font-bold text-gray-900 uppercase tracking-wider border-b border-brand-red/20" />,
  td: ({node, ...props}: any) => <td {...props} className="px-6 py-4 whitespace-nowrap text-sm text-gray-700 border-b" />,
  code: ({node, ...props}: any) => {
      const { className, children, ...rest } = props;
      const match = /language-(\w+)/.exec(className || '');
      const isBlock = !!match;

      return isBlock ? (
        <pre className="bg-[#1e1e1e] text-gray-100 p-4 rounded-lg overflow-x-auto my-4 text-sm font-mono leading-normal shadow-inner border-l-4 border-brand-gold">
           <code className={className} {...rest}>
            {children}
           </code>
        </pre>
      ) : (
        <code className={`bg-gray-100 text-brand-red px-1.5 py-0.5 rounded text-sm font-mono border border-gray-200 ${className || ''}`} {...rest}>
          {children}
        </code>
      )
  },
  blockquote: ({node, ...props}: any) => <blockquote {...props} className="border-l-4 border-brand-gold pl-4 py-1 my-4 italic text-gray-700 bg-gray-50 rounded-r" />,
  ul: ({node, ...props}: any) => <ul {...props} className="list-disc pl-6 space-y-1 my-4 marker:text-brand-red" />,
  ol: ({node, ...props}: any) => <ol {...props} className="list-decimal pl-6 space-y-1 my-4 marker:text-brand-red" />,
  h1: ({node, ...props}: any) => <h1 {...props} className="text-3xl font-bold text-black border-b border-gray-200 pb-2 mt-8 mb-4" />,
  h2: ({node, ...props}: any) => <h2 {...props} className="text-2xl font-bold text-gray-900 mt-8 mb-4" />,
  h3: ({node, ...props}: any) => <h3 {...props} className="text-xl font-bold text-gray-800 mt-6 mb-3" />,
  h4: ({node, ...props}: any) => <h4 {...props} className="text-lg font-bold text-gray-800 mt-6 mb-3" />,
  h5: ({node, ...props}: any) => <h5 {...props} className="text-base font-bold text-gray-800 mt-4 mb-2" />,
  h6: ({node, ...props}: any) => <h6 {...props} className="text-sm font-bold text-gray-800 mt-4 mb-2 uppercase tracking-wide text-brand-red" />,
  hr: ({node, ...props}: any) => <hr {...props} className="my-8 border-gray-300" />,
  p: ({node, ...props}: any) => <p {...props} className="mb-4 leading-relaxed text-gray-700" />,
};

function App() {
  const [doc, setDoc] = useState<DocumentState>({
    title: 'Untitled Document',
    content: defaultMarkdown
  });
  const [viewMode, setViewMode] = useState<ViewMode>('split');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const editorContainerRef = useRef<HTMLDivElement>(null);
  const previewContainerRef = useRef<HTMLDivElement>(null);

  // Sync scroll between editor and preview
  const handleScroll = (source: 'editor' | 'preview') => {
    const editor = editorContainerRef.current;
    const preview = previewContainerRef.current;
    
    if (!editor || !preview) return;

    if (source === 'editor') {
      const percentage = editor.scrollTop / (editor.scrollHeight - editor.clientHeight);
      preview.scrollTop = percentage * (preview.scrollHeight - preview.clientHeight);
    } else {
      const percentage = preview.scrollTop / (preview.scrollHeight - preview.clientHeight);
      editor.scrollTop = percentage * (editor.scrollHeight - editor.clientHeight);
    }
  };

  const handleInsert = (type: InsertType) => {
    if (!textareaRef.current) return;

    const textarea = textareaRef.current;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;

    const { text, newCursor } = insertMarkdown(doc.content, type, start, end);
    
    setDoc(prev => ({ ...prev, content: text }));

    setTimeout(() => {
      textarea.focus();
      textarea.setSelectionRange(newCursor, newCursor);
    }, 0);
  };

  const handleExportMarkdown = () => {
    const blob = new Blob([doc.content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${doc.title || 'untitled'}.md`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleExportHTML = () => {
    const htmlContent = `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${doc.title}</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/github-markdown-css@5/github-markdown.min.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    body {
        font-family: 'Inter', sans-serif;
        background-color: #ffffff;
        margin: 0;
        padding: 0;
    }
    .markdown-body {
      box-sizing: border-box;
      min-width: 200px;
      max-width: 980px;
      margin: 40px auto;
      padding: 45px;
      background-color: white;
      border-radius: 8px;
    }
    .markdown-body a {
        color: #DC2626;
        text-decoration: none;
    }
    .markdown-body a:hover {
        text-decoration: underline;
    }
    @media (max-width: 767px) {
      .markdown-body {
        padding: 15px;
        margin: 0;
        border-radius: 0;
      }
    }
  </style>
</head>
<body>
  <article class="markdown-body">
    ${document.getElementById('markdown-preview')?.innerHTML || ''}
  </article>
</body>
</html>`;

    const blob = new Blob([htmlContent], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${doc.title || 'untitled'}.html`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleExportPDF = () => {
    // Direct call to window.print()
    // The specific 'print:block' and 'print:hidden' classes in the JSX below
    // handle the visibility of what gets printed.
    window.print();
  };

  return (
    <div className="h-screen bg-gray-100 flex flex-col overflow-hidden">
      {/* Screen View (Hidden when printing) */}
      <div className="flex flex-col h-full print:hidden">
        <Header 
            title={doc.title}
            onTitleChange={(title) => setDoc(prev => ({ ...prev, title }))}
            onExportMarkdown={handleExportMarkdown}
            onExportHTML={handleExportHTML}
            onExportPDF={handleExportPDF}
        />
        
        <div className="flex flex-col flex-1 overflow-hidden relative shadow-inner">
            <Toolbar 
            onInsert={handleInsert} 
            viewMode={viewMode}
            onViewModeChange={setViewMode}
            />
            
            <div className="flex flex-1 overflow-hidden">
            <div 
                ref={editorContainerRef}
                className={`flex flex-col h-full overflow-y-auto border-r border-gray-200 bg-white transition-all duration-300 ${
                viewMode === 'editor' ? 'w-full' : 
                viewMode === 'split' ? 'w-1/2' : 'w-0 hidden'
                }`}
                onScroll={() => viewMode === 'split' && handleScroll('editor')}
            >
                <div className="flex-1 relative min-h-full">
                <textarea
                    ref={textareaRef}
                    value={doc.content}
                    onChange={(e) => setDoc(prev => ({ ...prev, content: e.target.value }))}
                    className="w-full h-full p-8 font-mono text-sm leading-relaxed resize-none outline-none text-gray-800 placeholder-gray-300"
                    placeholder="Start writing..."
                    spellCheck={false}
                />
                </div>
            </div>

            <div 
                ref={previewContainerRef}
                className={`flex flex-col h-full bg-white overflow-y-auto transition-all duration-300 ${
                viewMode === 'preview' ? 'w-full' : 
                viewMode === 'split' ? 'w-1/2' : 'w-0 hidden'
                }`}
                onScroll={() => viewMode === 'split' && handleScroll('preview')}
            >
                <div 
                id="markdown-preview" 
                className="prose prose-red max-w-none p-8 md:p-12 markdown-body"
                >
                <ReactMarkdown 
                    remarkPlugins={[remarkGfm, remarkMath]}
                    rehypePlugins={[rehypeKatex]}
                    components={markdownComponents}
                >
                    {doc.content}
                </ReactMarkdown>
                </div>
            </div>
            </div>
        </div>
      </div>

      {/* Print View (Only visible when printing) */}
      <div className="hidden print:block bg-white h-auto w-full p-8">
         <div className="prose prose-red max-w-none" style={{ maxWidth: 'none' }}>
             <h1 className="text-4xl font-bold mb-6 pb-2 border-b-2 border-gray-300">{doc.title}</h1>
             <div className="markdown-content">
               <ReactMarkdown 
                  remarkPlugins={[remarkGfm, remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                  components={markdownComponents}
              >
                  {doc.content}
              </ReactMarkdown>
             </div>
         </div>
      </div>
    </div>
  );
}

export default App;