import { createBrowserRouter } from 'react-router-dom';
import { Layout } from './components/Layout';
import { BenchmarkPage } from './pages/BenchmarkPage';
import { DownloadsPage } from './pages/DownloadsPage';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      { index: true, element: <BenchmarkPage /> },
      { path: 'downloads', element: <DownloadsPage /> },
    ],
  },
]);
