import { Outlet } from 'react-router-dom';
import { NavBar } from './NavBar';

export function Layout() {
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <header>
        <NavBar />
      </header>
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <Outlet />
      </main>
      <footer className="border-t border-gray-800 py-6 text-center text-sm text-gray-500">
        <p>
          <a
            href="https://github.com/quebi-gmbh/equana-benchmark"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-400 transition-colors hover:text-blue-400"
          >
            github.com/quebi-gmbh/equana-benchmark
          </a>
        </p>
      </footer>
    </div>
  );
}
