import { RadioGroup, Radio, Label } from 'react-aria-components';

const THREAD_OPTIONS = [1, 2, 4, 8, 16, 32] as const;

interface ThreadCountPickerProps {
  value: number;
  onChange: (threads: number) => void;
  isDisabled?: boolean;
}

export function ThreadCountPicker({ value, onChange, isDisabled }: ThreadCountPickerProps) {
  const hwThreads = typeof navigator !== 'undefined' ? (navigator.hardwareConcurrency ?? 0) : 0;

  return (
    <RadioGroup
      value={String(value)}
      onChange={(v) => onChange(Number(v))}
      isDisabled={isDisabled}
      orientation="horizontal"
      className="flex flex-col gap-1.5"
    >
      <Label className="text-xs font-medium tracking-wide text-gray-400 uppercase">
        Threads (MT){hwThreads > 0 && <span className="normal-case text-gray-500"> — {hwThreads} available</span>}
      </Label>
      <div className="flex flex-wrap gap-1">
        {THREAD_OPTIONS.map((n) => (
          <Radio
            key={n}
            value={String(n)}
            className="cursor-pointer rounded-md border border-gray-700 px-3 py-1.5 text-sm font-mono text-gray-400 transition-colors
              data-[selected]:border-emerald-500 data-[selected]:bg-emerald-500 data-[selected]:text-gray-950 data-[selected]:font-semibold
              data-[hovered]:bg-gray-800
              data-[focus-visible]:ring-2 data-[focus-visible]:ring-emerald-400 data-[focus-visible]:ring-offset-1 data-[focus-visible]:ring-offset-gray-950
              data-[disabled]:opacity-50 data-[disabled]:cursor-not-allowed"
          >
            {n}
          </Radio>
        ))}
      </div>
    </RadioGroup>
  );
}
