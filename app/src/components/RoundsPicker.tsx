import { RadioGroup, Radio, Label } from 'react-aria-components';

const ROUNDS = [1, 3, 5] as const;

interface RoundsPickerProps {
  value: number;
  onChange: (rounds: number) => void;
  isDisabled?: boolean;
}

export function RoundsPicker({ value, onChange, isDisabled }: RoundsPickerProps) {
  return (
    <RadioGroup
      value={String(value)}
      onChange={(v) => onChange(Number(v))}
      isDisabled={isDisabled}
      orientation="horizontal"
      className="flex flex-col gap-1.5"
    >
      <Label className="text-xs font-medium tracking-wide text-gray-400 uppercase">
        Rounds
      </Label>
      <div className="flex gap-1">
        {ROUNDS.map((r) => (
          <Radio
            key={r}
            value={String(r)}
            className="cursor-pointer rounded-md border border-gray-700 px-3 py-1.5 text-sm font-mono text-gray-400 transition-colors
              data-[selected]:border-blue-500 data-[selected]:bg-blue-500 data-[selected]:text-gray-950 data-[selected]:font-semibold
              data-[hovered]:bg-gray-800
              data-[focus-visible]:ring-2 data-[focus-visible]:ring-blue-400 data-[focus-visible]:ring-offset-1 data-[focus-visible]:ring-offset-gray-950
              data-[disabled]:opacity-50 data-[disabled]:cursor-not-allowed"
          >
            {r}
          </Radio>
        ))}
      </div>
    </RadioGroup>
  );
}
