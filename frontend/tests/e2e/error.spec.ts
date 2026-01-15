import { test, expect } from '@playwright/test';

test('selecting an invalid file shows client-side alert', async ({ page }) => {
  page.on('dialog', async (dialog) => {
    expect(dialog.message()).toContain('Please select a valid image file.');
    await dialog.dismiss();
  });

  await page.goto('/');
  const input = page.locator('input[type="file"]');
  await input.setInputFiles('tests/e2e/assets/sample_bad.txt');
});
