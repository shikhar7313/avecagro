import express from 'express'
import cors from 'cors'
import { fileURLToPath } from 'url'
import path from 'path'
import fs from 'fs/promises'
 
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// Prefer shared data file; fallback to assets copy if present
const DATA_PATH_CANDIDATES = [
  path.resolve(__dirname, '..', '..', 'data', 'questionaire.json'),
  path.resolve(__dirname, '..', 'Assets', 'questionaire.json'),
]

let resolvedDataPath = null

async function getDataPath() {
  if (resolvedDataPath) return resolvedDataPath

  for (const candidate of DATA_PATH_CANDIDATES) {
    try {
      await fs.access(candidate)
      resolvedDataPath = candidate
      return resolvedDataPath
    } catch {
      // try next
    }
  }

  resolvedDataPath = DATA_PATH_CANDIDATES[0]
  return resolvedDataPath
}
const PORT = Number(process.env.PORT ?? 7001)
 
const app = express()
 
app.use(cors({
  origin: [
    'http://localhost:3000',
    'http://localhost:5173',
    'http://localhost:6001',
    'http://localhost:6002',
    'http://localhost:7001',
  ],
}))
app.use(express.json({ limit: '2mb' }))
 
async function ensureDataFile() {
  try {
    const dataPath = await getDataPath()
    const directory = path.dirname(dataPath)
    await fs.mkdir(directory, { recursive: true })

    const buffer = await fs.readFile(dataPath, 'utf8').catch(() => '[]')
    const parsed = JSON.parse(buffer)
    if (Array.isArray(parsed)) {
      return { entries: parsed, dataPath }
    }
    return { entries: [], dataPath }
  } catch (error) {
    console.error('Failed to read questionnaire file', error)
    return { entries: [], dataPath: await getDataPath() }
  }
}
 
async function saveEntries(entries) {
  const serialized = JSON.stringify(entries, null, 2)
  const dataPath = await getDataPath()
  await fs.writeFile(dataPath, serialized, 'utf8')
}
 
app.get('/health', async (_req, res) => {
  const { entries, dataPath } = await ensureDataFile()
  res.json({ status: 'ok', entries: entries.length, path: dataPath })
})
 
app.get('/api/questionnaire', async (_req, res) => {
  const { entries } = await ensureDataFile()
  res.json({ ok: true, entries })
})
 
app.get('/api/questionnaire/latest', async (req, res) => {
  const { entries } = await ensureDataFile()
  const { username } = req.query
 
  let filtered = entries
  if (username) {
    filtered = entries.filter(entry => String(entry.username) === String(username))
  }
 
  if (!filtered.length) {
    return res.json({ ok: true, entry: null })
  }
 
  const latest = filtered.reduce((prev, current) => {
    const prevTime = new Date(prev.savedAt || prev.confirmedAt || prev.id).getTime()
    const currentTime = new Date(current.savedAt || current.confirmedAt || current.id).getTime()
    return currentTime > prevTime ? current : prev
  })
 
  res.json({ ok: true, entry: latest })
})
 
app.post('/api/questionnaire', async (req, res) => {
  const { username, answers, selectedCrop = null, recommendations = [], planId } = req.body ?? {}
 
  if (!username || typeof answers !== 'object') {
    return res.status(400).json({
      ok: false,
      message: 'username and answers are required',
    })
  }
 
  const entry = {
    id: Date.now(),
    username,
    planId: planId ?? null,
    answers,
    selectedCrop,
    recommendations,
    savedAt: new Date().toISOString(),
    confirmedAt: selectedCrop ? new Date().toISOString() : null,
  }
 
  try {
    const { entries } = await ensureDataFile()
    entries.push(entry)
    await saveEntries(entries)
    res.status(201).json({ ok: true, entry })
  } catch (error) {
    console.error('Failed to persist questionnaire', error)
    res.status(500).json({ ok: false, message: 'Unable to save questionnaire' })
  }
})
 
app.patch('/api/questionnaire/:id/selection', async (req, res) => {
  const { id } = req.params
  const { selectedCrop, recommendations = [] } = req.body ?? {}
 
  if (!selectedCrop) {
    return res.status(400).json({ ok: false, message: 'selectedCrop is required' })
  }
 
  try {
    const { entries } = await ensureDataFile()
    const index = entries.findIndex(entry => String(entry.id) === String(id))
    if (index === -1) {
      return res.status(404).json({ ok: false, message: 'Entry not found' })
    }
 
    entries[index].selectedCrop = selectedCrop
    entries[index].recommendations = recommendations
    entries[index].confirmedAt = new Date().toISOString()
 
    await saveEntries(entries)
    res.json({ ok: true, entry: entries[index] })
  } catch (error) {
    console.error('Failed to update questionnaire selection', error)
    res.status(500).json({ ok: false, message: 'Unable to update selection' })
  }
})
 
app.listen(PORT, () => {
  console.log(`Avec Agro questionnaire server listening on http://localhost:${PORT}`)
})
 
 