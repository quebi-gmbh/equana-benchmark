import { Button } from 'react-aria-components';

interface RunAllButtonProps {
  onPress: () => void;
  isRunning: boolean;
  progress?: { current: number; total: number };
}

export function RunAllButton({ onPress, isRunning, progress }: RunAllButtonProps) {
  return (
    <Button
      onPress={onPress}
      isDisabled={isRunning}
      className="rounded-lg bg-blue-500 px-5 py-2 text-sm font-semibold text-gray-950 transition-all
        hover:bg-blue-400 hover:shadow-[0_0_20px_rgba(59,130,246,0.3)]
        data-[pressed]:bg-blue-600
        data-[disabled]:bg-gray-700 data-[disabled]:text-gray-500 data-[disabled]:cursor-not-allowed data-[disabled]:shadow-none
        data-[focus-visible]:ring-2 data-[focus-visible]:ring-blue-400 data-[focus-visible]:ring-offset-2 data-[focus-visible]:ring-offset-gray-950"
    >
      {isRunning && progress
        ? `Running... (${progress.current}/${progress.total})`
        : isRunning
          ? 'Running...'
          : 'Run All'}
    </Button>
  );
}
