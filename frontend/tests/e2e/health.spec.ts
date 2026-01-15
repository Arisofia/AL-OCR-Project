import { test, expect } from '@playwright/test';

test('shows healthy status badge', async ({ page }) => {
  await page.goto('/');
  // Status badge contains text like "System healthy"
  const badge = page.locator('.status-badge');
  await expect(badge).toHaveText(/System\s+healthy/);
});
