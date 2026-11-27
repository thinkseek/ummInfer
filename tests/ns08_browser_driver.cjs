#!/usr/bin/env node
/* NS-08 real Chromium default-DOM probe; no model or resident-service contact. */
'use strict';

const fs = require('fs');
const path = require('path');

async function main() {
  const [pagePath, reportPath, screenshotPath, playwrightCore, chromiumPath, controlsJson] =
    process.argv.slice(2);
  if (!pagePath || !reportPath || !screenshotPath || !playwrightCore ||
      !chromiumPath || !controlsJson) {
    throw new Error('usage: ns08_browser_driver.cjs PAGE REPORT SCREENSHOT PLAYWRIGHT_CORE CHROMIUM CONTROLS_JSON');
  }
  const controls = JSON.parse(controlsJson);
  const groups = {
    visible_modality_control_count: ['image', 'audio', 'video'],
    visible_routing_control_count: ['narrative', 'visualMode', 'maxSlots', 'allowTextOnly',
      'narrativeContinuity', 'styleAnchor', 'fastVideo'],
    visible_sampling_control_count: ['maxTokens', 'temperature', 'topP', 'seed',
      'imageSteps', 'videoSteps', 'videoGuidance'],
    visible_frame_control_count: ['videoFramePolicy', 'videoFrameBudget',
      'videoMotionThreshold', 'videoSceneThreshold'],
    visible_voice_routing_control_count: ['voiceToggle', 'voice', 'realtimeEchoSuppression'],
  };
  const result = {status: 'FAIL', checks: [], actions: [], network_requests: [],
                  visible_controls: []};
  const check = (name, ok, detail = null) => result.checks.push({name, ok: Boolean(ok), detail});
  const {chromium} = require(path.resolve(playwrightCore));
  const browser = await chromium.launch({headless: true, executablePath: path.resolve(chromiumPath),
                                         args: ['--no-sandbox']});
  try {
    const context = await browser.newContext({ignoreHTTPSErrors: false});
    const page = await context.newPage();
    page.on('request', request => {
      if (/^https?:/.test(request.url())) result.network_requests.push(request.url());
    });
    await page.goto('file://' + path.resolve(pagePath), {waitUntil: 'load'});
    await page.locator('#prompt').waitFor({state: 'visible'});
    const advancedOpen = await page.locator('#advanced').evaluate(element => element.open);
    check('Advanced initially collapsed', advancedOpen === false);
    const visible = await page.evaluate(ids => ids.filter(id => {
      const element = document.getElementById(id);
      if (!element) return false;
      const style = getComputedStyle(element);
      const rendered = typeof element.checkVisibility !== 'function' ||
        element.checkVisibility({checkOpacity: true, checkVisibilityCSS: true});
      return style.display !== 'none' && style.visibility !== 'hidden' && rendered &&
             element.getClientRects().length > 0;
    }), controls);
    result.visible_controls = [...visible].sort();
    result.visible_counts = {};
    for (const [name, ids] of Object.entries(groups)) {
      result.visible_counts[name] = ids.filter(id => visible.includes(id)).length;
    }
    check('zero default developer controls', visible.length === 0, visible);
    for (const id of ['prompt', 'send', 'realtimeToggle', 'importSession', 'new']) {
      check(`natural control ${id} remains visible`, await page.locator('#' + id).isVisible());
    }
    check('default JSON export is lossless', await page.locator('#exportFormat').inputValue() === 'replay');
    await page.locator('#prompt').focus();
    result.startup_network_requests = [...result.network_requests];
    result.network_requests.length = 0;
    await page.keyboard.press('Control+Enter');
    check('empty keyboard send creates no network request', result.network_requests.length === 0,
          result.network_requests);
    await page.keyboard.press('Alt+N');
    check('New chat keyboard shortcut keeps Advanced collapsed',
          await page.locator('#advanced').evaluate(element => element.open) === false);
    result.actions.push('Ctrl+Enter with empty prompt; Alt+N; no Advanced interaction');
    await page.screenshot({path: path.resolve(screenshotPath), fullPage: true});
    result.status = result.checks.every(row => row.ok) ? 'PASS' : 'FAIL';
    await context.close();
  } finally {
    await browser.close();
  }
  result.checks_failed = result.checks.filter(row => !row.ok).length;
  result.checks_passed = result.checks.filter(row => row.ok).length;
  fs.writeFileSync(reportPath, JSON.stringify(result, null, 2) + '\n');
}

main().catch(error => {
  console.error(error && error.stack ? error.stack : String(error));
  process.exitCode = 2;
});
