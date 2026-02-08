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
      <TabList className="flex border-b border-gray-700/50">
        <Tab id="ubuntu" className={tabClass}>Ubuntu / Debian</Tab>
        <Tab id="macos" className={tabClass}>macOS</Tab>
        <Tab id="windows" className={tabClass}>Windows</Tab>
      </TabList>

      <TabPanel id="ubuntu" className="space-y-4">
        <p className="text-sm text-gray-400">
          Clone the repo and run the benchmark scripts directly:
        </p>
        <CodeBlock language="bash">{`# Install dependencies
sudo apt update
sudo apt install -y build-essential gfortran

# Clone the repository
git clone https://github.com/quebi-gmbh/equana-benchmark.git
cd equana-benchmark/matmul-benchmarks

# 1. NumPy/OpenBLAS benchmark
pip install numpy
python run_numpy_benchmarks.py

# 2. MKL benchmark (direct ctypes, no NumPy needed)
pip install mkl
python run_mkl_benchmarks.py

# 3. Native C / OpenBLAS benchmark (builds OpenBLAS from source)
cd native-openblas
bash build_all.sh    # ~10 min, compiles 4 architecture variants
bash run_benchmarks.sh
cd ..

# 4. MATLAB benchmark (requires MATLAB license)
bash run_matlab_benchmarks.sh`}</CodeBlock>
      </TabPanel>

      <TabPanel id="macos" className="space-y-4">
        <p className="text-sm text-gray-400">
          On macOS, install dependencies via Homebrew. The NumPy and MATLAB benchmarks work out of the box.
          The native C/OpenBLAS build compiles from source and should work on both Apple Silicon and Intel Macs.
        </p>
        <CodeBlock language="bash">{`# Install dependencies
brew install gcc gfortran

# Clone the repository
git clone https://github.com/quebi-gmbh/equana-benchmark.git
cd equana-benchmark/matmul-benchmarks

# 1. NumPy/OpenBLAS benchmark
pip3 install numpy
python3 run_numpy_benchmarks.py

# 2. Native C / OpenBLAS benchmark
cd native-openblas
bash build_all.sh
bash run_benchmarks.sh
cd ..

# 3. MATLAB benchmark (requires MATLAB license)
bash run_matlab_benchmarks.sh`}</CodeBlock>
      </TabPanel>

      <TabPanel id="windows" className="space-y-4">
        <p className="text-sm text-gray-400">
          On Windows, use WSL2 (recommended) or run individual benchmarks natively:
        </p>

        <h4 className="text-sm font-semibold text-gray-300">Option 1: WSL2 (Recommended)</h4>
        <CodeBlock language="bash">{`# Install WSL2 with Ubuntu (from PowerShell as admin)
wsl --install

# Then follow the Ubuntu instructions above inside WSL2`}</CodeBlock>

        <h4 className="text-sm font-semibold text-gray-300">Option 2: Native Python + MATLAB</h4>
        <p className="text-sm text-gray-400">
          The NumPy and MATLAB benchmarks can run natively on Windows. The native C/OpenBLAS build requires WSL2 or MSYS2.
        </p>
        <CodeBlock language="bash">{`# NumPy/OpenBLAS benchmark (works in native Windows Python)
pip install numpy
python run_numpy_benchmarks.py

# MKL benchmark (works in native Windows Python)
pip install mkl
python run_mkl_benchmarks.py

# MATLAB benchmark (works in native Windows MATLAB)
matlab -batch "run_matlab_benchmarks"`}</CodeBlock>
      </TabPanel>
    </Tabs>
  );
}
