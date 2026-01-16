import { test, expect } from '@playwright/test';

test('shows healthy status badge', async ({ page }) => {
  // Mock health check success
  await page.route('**/health', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ status: 'healthy', timestamp: Date.now() })
    });
  });

  await page.goto('/');
  // Status badge contains text like "System healthy"
  const badge = page.locator('.status-badge');
  await expect(badge).toHaveText(/System\s+healthy/);
});
