import { test, expect } from '@playwright/test';
import path from 'path';

test('upload and extract text flow', async ({ page, baseURL }) => {
  await page.goto('/');

  // Ensure the file input is present
  const fileInput = await page.locator('input[type=file]');
  const samplePath = path.resolve(process.cwd(), '../../ocr_reconstruct/tests/data/sample_pixelated.png');

  // Set file and preview
  await fileInput.setInputFiles(samplePath);
  await expect(page.getByAltText('Preview')).toBeVisible({ timeout: 5000 });

  // Click extract and wait for result
  const button = page.getByRole('button', { name: /Extract Data/i });
  await button.click();

  // Wait for processing indicator and then result
  await page.waitForSelector('.ocr-text', { timeout: 20000 });
  const resultText = await page.locator('.ocr-text').innerText();
  expect(resultText.length).toBeGreaterThan(0);
});
