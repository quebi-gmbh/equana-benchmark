import { useState } from 'react';
import { Button } from 'react-aria-components';

interface CodeBlockProps {
  children: string;
  language?: string;
}

export function CodeBlock({ children, language }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(children);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="group relative rounded-lg border border-gray-600/50 bg-gray-900">
      {language && (
        <div className="border-b border-gray-600/50 px-3 py-1.5 text-xs font-medium text-gray-500">
          {language}
        </div>
      )}
      <pre className="overflow-x-auto p-4 text-sm leading-relaxed text-gray-300">
        <code>{children}</code>
      </pre>
      <Button
        onPress={() => void handleCopy()}
        className="absolute right-2 top-2 rounded-md bg-gray-800 px-2 py-1 text-xs text-gray-400 opacity-0 transition-opacity
          group-hover:opacity-100
          hover:bg-gray-700 hover:text-white
          data-[focus-visible]:opacity-100 data-[focus-visible]:ring-2 data-[focus-visible]:ring-blue-400"
      >
        {copied ? 'Copied!' : 'Copy'}
      </Button>
    </div>
  );
}
