import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Built into the backend so one `uvicorn` command serves everything at /dashboard/.
// `npm run dev` proxies the API to a running backend on :8000.
export default defineConfig({
  plugins: [react()],
  base: '/dashboard/',
  build: { outDir: '../backend/app/static/dashboard', emptyOutDir: true, chunkSizeWarningLimit: 1200 },
  server: { proxy: { '/api': 'http://localhost:8000', '/ws': { target: 'ws://localhost:8000', ws: true } } },
})
