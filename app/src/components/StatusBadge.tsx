type Status = 'idle' | 'running' | 'done' | 'error';

interface StatusBadgeProps {
  status: Status;
}

export function StatusBadge({ status }: StatusBadgeProps) {
  switch (status) {
    case 'idle':
      return <span className="inline-block h-2 w-2 rounded-full bg-gray-600" />;
    case 'running':
      return <span className="inline-block h-2 w-2 rounded-full bg-blue-400 animate-pulse shadow-[0_0_6px_rgba(59,130,246,0.6)]" />;
    case 'done':
      return <span className="inline-block h-2 w-2 rounded-full bg-emerald-400" />;
    case 'error':
      return <span className="inline-block h-2 w-2 rounded-full bg-red-400" />;
  }
}
