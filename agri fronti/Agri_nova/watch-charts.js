const path = require('path');
const chokidar = require('chokidar');
const express = require('express');
const { generateCharts } = require('./generate-charts');

const DATA_FILE = process.env.CHART_DATA_PATH || path.join(__dirname, 'src/data/dashboard/chartsDemoData.json');
const PORT = Number(process.env.CHART_SERVER_PORT || 6100);

const app = express();
let runState = {
  running: false,
  queued: false,
  lastRun: null,
  lastReason: null,
  lastError: null,
};

async function runGeneration(reason = 'manual') {
  if (runState.running) {
    runState.queued = true;
    return;
  }

  runState.running = true;
  runState.lastReason = reason;
  console.log(`[chart-watcher] Triggered (${reason}).`);

  try {
    const success = await generateCharts();
    if (!success) {
      runState.lastError = 'Chart generation returned false';
      console.error('[chart-watcher] Chart generation reported failure.');
    } else {
      runState.lastError = null;
      runState.lastRun = new Date().toISOString();
      console.log('[chart-watcher] Charts regenerated successfully.');
    }
  } catch (error) {
    runState.lastError = error.message;
    console.error('[chart-watcher] Chart generation failed:', error);
  } finally {
    runState.running = false;
    if (runState.queued) {
      runState.queued = false;
      runGeneration('queued');
    }
  }
}

function startWatcher() {
  const watcher = chokidar.watch(DATA_FILE, { ignoreInitial: false });
  watcher.on('add', () => runGeneration('file-added'));
  watcher.on('change', () => runGeneration('file-changed'));
  watcher.on('error', (error) => console.error('[chart-watcher] Watcher error:', error));
  console.log(`[chart-watcher] Watching ${DATA_FILE}`);
}

app.get('/status', (_req, res) => {
  res.json({
    watching: DATA_FILE,
    running: runState.running,
    lastRun: runState.lastRun,
    lastReason: runState.lastReason,
    lastError: runState.lastError,
  });
});

app.post('/trigger', async (_req, res) => {
  await runGeneration('http-trigger');
  res.json({ ok: true });
});

app.listen(PORT, () => {
  console.log(`[chart-watcher] Status server listening on http://localhost:${PORT}`);
});

startWatcher();
runGeneration('startup');
