import { test, expect } from '@playwright/test';
import path from 'path';

test.describe('OCR Flow', () => {
  test('should display healthy status and handle document upload', async ({ page }) => {
    // 1. Mock health check success
    await page.route('**/health', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ status: 'healthy', timestamp: Date.now() })
      });
    });

    // Navigate to the app
    await page.goto('/');

    // Verify health status
    const statusBadge = page.locator('.status-badge');
    await expect(statusBadge).toContainText('System healthy', { timeout: 10000 });

    // 2. Mock the OCR API response to avoid S3 dependency in E2E
    await page.route('**/ocr', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          text: 'EXTRACTED FINANCIAL DATA\nInvoice #12345\nAmount: $500.00',
          iterations: [
            { iteration: 1, text_length: 20 },
            { iteration: 2, text_length: 45 }
          ],
          processing_time: 0.85
        })
      });
    });

    // 3. Upload a file
    // We'll create a dummy file for the upload
    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.locator('input[type="file"]').click();
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles({
      name: 'test-invoice.png',
      mimeType: 'image/png',
      buffer: Buffer.from('fake-image-content')
    });

    // 4. Click Extract Data
    const extractButton = page.getByRole('button', { name: /Extract Data/i });
    await expect(extractButton).toBeEnabled();
    await extractButton.click();

    // 5. Verify results are displayed
    const resultSection = page.locator('.result-section');
    await expect(resultSection).toContainText('EXTRACTED FINANCIAL DATA');
    await expect(resultSection).toContainText('Invoice #12345');
    await expect(resultSection).toContainText('Amount: $500.00');

    // 6. Verify iteration metadata
    await expect(page.locator('.iteration-pills')).toBeVisible();
    await expect(page.locator('.iteration-pills')).toContainText('Iteration 1');
    await expect(page.locator('.iteration-pills')).toContainText('Iteration 2');
  });

  test('should show error when backend is offline', async ({ page }) => {
    // Mock health check failure
    await page.route('**/health', async (route) => {
      await route.fulfill({ status: 500, body: JSON.stringify({ status: 'unhealthy' }) });
    });

    await page.goto('/');

    const statusBadge = page.locator('.status-badge');
    await expect(statusBadge).toContainText('System offline', { timeout: 10000 });

    const extractButton = page.getByRole('button', { name: /Extract Data/i });
    await expect(extractButton).toBeDisabled();
  });
});
