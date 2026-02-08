import { RadioGroup, Radio, Label } from 'react-aria-components';

const SIZES = [128, 256, 512, 750, 1000, 1500, 2000, 4000] as const;

interface MatrixSizePickerProps {
  value: number;
  onChange: (size: number) => void;
  isDisabled?: boolean;
}

export function MatrixSizePicker({ value, onChange, isDisabled }: MatrixSizePickerProps) {
  return (
    <RadioGroup
      value={String(value)}
      onChange={(v) => onChange(Number(v))}
      isDisabled={isDisabled}
      orientation="horizontal"
      className="flex flex-col gap-1.5"
    >
      <Label className="text-xs font-medium tracking-wide text-gray-400 uppercase">
        Matrix Size (N x N)
      </Label>
      <div className="flex flex-wrap gap-1">
        {SIZES.map((size) => (
          <Radio
            key={size}
            value={String(size)}
            className="cursor-pointer rounded-md border border-gray-700 px-3 py-1.5 text-sm font-mono text-gray-400 transition-colors
              data-[selected]:border-blue-500 data-[selected]:bg-blue-500 data-[selected]:text-gray-950 data-[selected]:font-semibold
              data-[hovered]:bg-gray-800
              data-[focus-visible]:ring-2 data-[focus-visible]:ring-blue-400 data-[focus-visible]:ring-offset-1 data-[focus-visible]:ring-offset-gray-950
              data-[disabled]:opacity-50 data-[disabled]:cursor-not-allowed"
          >
            {size}
          </Radio>
        ))}
      </div>
    </RadioGroup>
  );
}
