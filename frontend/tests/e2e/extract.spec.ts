import { test, expect } from '@playwright/test';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test('upload and extract text flow', async ({ page, baseURL }) => {
  // Mock health check success
  await page.route('**/health', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ status: 'healthy', timestamp: Date.now() })
    });
  });

  // Mock the OCR API response
  await page.route('**/ocr', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        text: 'MOCKED EXTRACTED DATA',
        iterations: [{ iteration: 1, text_length: 20 }],
        processing_time: 0.5
      })
    });
  });

  await page.goto('/');

  // Ensure the file input is present
  const fileInput = await page.locator('input[type=file]');
  const samplePath = path.resolve(__dirname, '../../../ocr_reconstruct/tests/data/sample_pixelated.png');

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
