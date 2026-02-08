import { createServer } from 'http';
import { readFile } from 'fs/promises';
import { extname, join } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PORT = 3000;

const MIME_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8',
    '.mjs': 'application/javascript; charset=utf-8',
    '.wasm': 'application/wasm',
    '.css': 'text/css; charset=utf-8',
};

const server = createServer(async (req, res) => {
    // Use index-new.html as default (cleaner version)
    let filePath = req.url === '/' ? '/index-new.html' : req.url;
    filePath = join(__dirname, filePath);

    const ext = extname(filePath);
    const contentType = MIME_TYPES[ext] || 'application/octet-stream';

    try {
        const content = await readFile(filePath);
        res.writeHead(200, {
            'Content-Type': contentType,
            'Cross-Origin-Opener-Policy': 'same-origin',
            'Cross-Origin-Embedder-Policy': 'require-corp',
        });
        res.end(content);
    } catch (err) {
        res.writeHead(404);
        res.end(`Not found: ${req.url}`);
    }
});

server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
    console.log('Open in Chrome/Firefox and use DevTools Performance tab to profile');
    console.log('Press Ctrl+C to stop');
});
