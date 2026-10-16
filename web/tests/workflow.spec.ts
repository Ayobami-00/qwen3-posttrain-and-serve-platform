import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';

test('public boundaries and owner CPU training-to-serving', async ({ page, context }) => {
  test.setTimeout(420_000);
  const name = `browser-cpu-${Date.now()}`;
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Build, measure, repeat.' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'New experiment' })).toBeDisabled();
  expect([401, 422]).toContain((await context.request.post('/api/v1/runs', { data: {} })).status());
  expect((await context.request.get('/api/v1/admin/budget')).status()).toBe(401);
  if (process.env.QPS_LIVE_OWNER_FILE) {
    const owner = JSON.parse(readFileSync(process.env.QPS_LIVE_OWNER_FILE, 'utf8'));
    await page.getByRole('link', { name: 'Owner sign in' }).click();
    await page.locator('input[name="username"]').first().fill(owner.username);
    await page.locator('input[name="password"]').first().fill(owner.password);
    await page.locator('input[type="submit"],button[type="submit"]').first().click();
    await expect(page).toHaveURL(process.env.QPS_BROWSER_URL + '/');
  } else {
    const login = await context.request.post('/api/v1/auth/development', { headers: { Authorization: 'Bearer browser-test-owner-token' } });
    expect(login.ok()).toBeTruthy();
  }
  await page.reload();
  await expect(page.getByText('Owner session')).toBeVisible();
  await page.getByRole('button', { name: 'New experiment' }).click();
  await page.getByLabel('Experiment name').fill(name);
  await page.getByLabel('Compute and fixture').selectOption('cpu');
  await page.getByRole('button', { name: 'Submit experiment' }).click();
  await expect(page.getByRole('heading', { name: name })).toBeVisible();
  await expect(page.getByText('succeeded', { exact: true }).first()).toBeVisible({ timeout: 240_000 });
  await page.screenshot({ path: '../.qps/dashboard-experiment.png', fullPage: true });
  await page.getByRole('button', { name: 'configuration', exact: true }).click();
  await expect(page.getByText('Immutable submitted configuration')).toBeVisible();
  await page.getByRole('button', { name: 'artifacts', exact: true }).click();
  await expect(page.getByText('merged model', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Deploy', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Generate', exact: true })).toBeEnabled({ timeout: 120_000 });
  await page.getByRole('button', { name: 'Generate', exact: true }).click();
  await expect(page.locator('.answer')).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole('button', { name: 'Generate', exact: true })).toBeEnabled();
  await page.screenshot({ path: '../.qps/dashboard-serving.png', fullPage: true });
  await page.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Restart', exact: true })).toBeVisible({ timeout: 30_000 });
  await context.clearCookies();
  await page.reload();
  await page.getByRole('button', { name: 'Deployments', exact: true }).click();
  await expect(page.getByText('Chat playground')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Stop', exact: true })).toHaveCount(0);
});
