import { test, expect } from '@playwright/test';
import path from 'path';

test('upload and extract text flow', async ({ page, baseURL }) => {
  await page.goto('/');

  // Ensure the file input is present
  const fileInput = await page.locator('input[type=file]');
  const samplePath = path.resolve(process.cwd(), '../../ocr-reconstruct/tests/data/sample_pixelated.png');

  // Set file and preview
  await fileInput.setInputFiles(samplePath);
  await expect(page.locator('text=Preview')).toBeVisible({ timeout: 5000 });

  // Click extract and wait for result
  const button = page.getByRole('button', { name: /Extract Text/i });
  await button.click();

  // Wait for processing indicator and then result
  await page.waitForSelector('.result pre', { timeout: 20000 });
  const resultText = await page.locator('.result pre').innerText();
  expect(resultText.length).toBeGreaterThan(0);
});
