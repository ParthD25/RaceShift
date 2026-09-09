import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Local-first: the dev server binds to loopback only and proxies /api to the FastAPI backend.
// Ports come from the environment so two checkouts can run side by side:
//   API_PORT=8010 WEB_PORT=5180 npm run dev
const apiPort = Number(process.env.API_PORT ?? 8000);
const webPort = Number(process.env.WEB_PORT ?? 5173);

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: webPort,
    strictPort: true,
    proxy: {
      '/api': `http://127.0.0.1:${apiPort}`
    }
  },
  preview: {
    host: '127.0.0.1',
    port: Number(process.env.PREVIEW_PORT ?? 4173),
    strictPort: true
  },
  build: {
    rollupOptions: {
      output: {
        // Keep the chart library in its own chunk so the app shell stays small.
        manualChunks: {
          charts: ['recharts'],
          react: ['react', 'react-dom', 'react-router-dom']
        }
      }
    }
  }
});
