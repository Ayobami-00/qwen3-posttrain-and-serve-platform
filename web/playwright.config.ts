import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests', timeout: 180_000, workers: 1,
  use: { baseURL: process.env.QPS_BROWSER_URL || 'http://127.0.0.1:3000', viewport: { width: 1440, height: 1000 }, screenshot: 'only-on-failure', launchOptions: { executablePath: process.env.QPS_CHROMIUM_EXECUTABLE || undefined } },
  reporter: [['list'], ['json', { outputFile: '../.qps/browser-results.json' }]],
});
