import express from 'express'
import cors from 'cors'
import { fileURLToPath } from 'url'
import path from 'path'
import fs from 'fs/promises'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const ROOT_DIR = path.resolve(__dirname, '..')
const DATA_PATH = path.join(ROOT_DIR, 'Assets', 'login.json')
const PORT = Number(process.env.PORT ?? 6002)

const app = express()

app.use(cors({
  origin: [
    'http://localhost:5173',
    'http://localhost:6001',
    'http://localhost:6002',
  ],
}))
app.use(express.json({ limit: '1mb' }))

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', target: DATA_PATH })
})

app.post('/api/users', async (req, res) => {
  const payload = req.body
  if (!Array.isArray(payload)) {
    return res.status(400).json({ ok: false, message: 'Expected an array payload.' })
  }
  try {
    const directory = path.dirname(DATA_PATH)
    await fs.mkdir(directory, { recursive: true })
    await fs.writeFile(DATA_PATH, JSON.stringify(payload, null, 2), 'utf8')
    res.json({ ok: true, saved: payload.length, path: DATA_PATH })
  } catch (error) {
    console.error('Failed to write login.json', error)
    res.status(500).json({ ok: false, message: 'Unable to persist users JSON.' })
  }
})

app.listen(PORT, () => {
  console.log(`Avec Agro save-users server listening on http://localhost:${PORT}`)
})
