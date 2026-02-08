import { Tabs, TabList, Tab, TabPanel } from 'react-aria-components';
import { CodeBlock } from './CodeBlock';

const tabClass = `px-4 py-2 text-sm font-medium cursor-pointer rounded-t-md transition-colors outline-none
  border-b-2 border-transparent text-gray-400
  data-[selected]:border-blue-400 data-[selected]:text-blue-400 data-[selected]:bg-gray-900/50
  data-[hovered]:text-gray-200
  data-[focus-visible]:ring-2 data-[focus-visible]:ring-blue-400`;

export function PlatformTabs() {
  return (
    <Tabs defaultSelectedKey="ubuntu" className="space-y-4">
      <TabList className="flex border-b border-gray-800">
        <Tab id="ubuntu" className={tabClass}>Ubuntu / Debian</Tab>
        <Tab id="macos" className={tabClass}>macOS</Tab>
        <Tab id="windows" className={tabClass}>Windows</Tab>
      </TabList>

      <TabPanel id="ubuntu" className="space-y-4">
        <p className="text-sm text-gray-400">
          The build scripts are designed for Ubuntu/Debian. Install dependencies and compile:
        </p>
        <CodeBlock language="bash">{`# Install dependencies
sudo apt update
sudo apt install -y build-essential libopenblas-dev

# Clone the repository
git clone https://github.com/quebi-gmbh/equana-benchmark.git
cd equana-benchmark/matmul-benchmarks

# Compile the native benchmark
gcc -O3 -march=native -o matmul_bench matmul_openblas.c \\
    -lopenblas -lpthread -lm

# Run
./matmul_bench`}</CodeBlock>

        <CodeBlock language="bash">{`# For the Python/NumPy benchmark
pip install numpy
python run_numpy_benchmarks.py`}</CodeBlock>
      </TabPanel>

      <TabPanel id="macos" className="space-y-4">
        <p className="text-sm text-gray-400">
          On macOS, install OpenBLAS via Homebrew and adjust the include/library paths:
        </p>
        <CodeBlock language="bash">{`# Install dependencies
brew install openblas

# Clone the repository
git clone https://github.com/quebi-gmbh/equana-benchmark.git
cd equana-benchmark/matmul-benchmarks

# Compile with Homebrew OpenBLAS paths
gcc -O3 -march=native -o matmul_bench matmul_openblas.c \\
    -I/opt/homebrew/opt/openblas/include \\
    -L/opt/homebrew/opt/openblas/lib \\
    -lopenblas -lpthread -lm

# On Intel Macs, use /usr/local instead of /opt/homebrew:
# -I/usr/local/opt/openblas/include
# -L/usr/local/opt/openblas/lib

# Run
./matmul_bench`}</CodeBlock>

        <CodeBlock language="bash">{`# For the Python/NumPy benchmark
pip3 install numpy
python3 run_numpy_benchmarks.py`}</CodeBlock>
      </TabPanel>

      <TabPanel id="windows" className="space-y-4">
        <p className="text-sm text-gray-400">
          On Windows, use WSL2 (recommended) or MSYS2/MinGW:
        </p>

        <h4 className="text-sm font-semibold text-gray-300">Option 1: WSL2 (Recommended)</h4>
        <CodeBlock language="bash">{`# Install WSL2 with Ubuntu (from PowerShell as admin)
wsl --install

# Then follow the Ubuntu instructions above inside WSL2`}</CodeBlock>

        <h4 className="text-sm font-semibold text-gray-300">Option 2: MSYS2 / MinGW</h4>
        <CodeBlock language="bash">{`# Install MSYS2 from https://www.msys2.org/
# Open MSYS2 MINGW64 terminal

# Install dependencies
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-openblas

# Clone and compile
git clone https://github.com/quebi-gmbh/equana-benchmark.git
cd equana-benchmark/matmul-benchmarks

gcc -O3 -march=native -o matmul_bench.exe matmul_openblas.c \\
    -lopenblas -lpthread

# Run
./matmul_bench.exe`}</CodeBlock>
      </TabPanel>
    </Tabs>
  );
}
