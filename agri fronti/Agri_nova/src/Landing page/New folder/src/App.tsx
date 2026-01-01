import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode, type FormEvent } from 'react'
import { Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import logoImage from '../Assets/Logo.png'
import seedUsers from './data/users.json'
import useWeatherLogic from './hooks/useWeatherLogic'

/* eslint-disable react-hooks/rules-of-hooks */

const trustLogos = ['IARI', 'ICAR', 'FPO Partners', 'State Agriculture Departments', 'Krishi Vigyan Kendra']

const heroStats = [
  { label: 'Pesticide Saved', value: '35%' },
  { label: 'Water Reduced', value: '22%' },
  { label: 'ROI Boost', value: '4.2×' },
]

const features = [
  {
    title: 'AI Crop Planning',
    description: 'Get crop recommendations based on soil health, season, and budget realities.',
  },
  {
    title: 'Daily Smart Tasks',
    description: 'Know exactly what to do today—irrigation, nutrition, protection, and labor.',
  },
  {
    title: 'Automated Field Monitoring',
    description: 'Drone scans, rover inspections, and soil sensors stream live insights.',
  },
]

const engines = [
  { title: 'Soil Intelligence Engine', copy: 'Reads your soil profile like a lab report.' },
  { title: 'Micro-Climate Engine', copy: 'Predicts stress pockets before they hit the crop.' },
  { title: 'Crop Recommendation', copy: 'Selects the right crop mix per season and market.' },
  { title: 'Task Engine', copy: 'Generates prioritized daily playbooks.' },
  { title: 'Pest Foresight', copy: 'Warns 7–10 days before a likely outbreak.' },
  { title: 'Vision Engine', copy: 'Drone and rover imagery detects hotspots & disease.' },
  { title: 'Yield Forecast Engine', copy: 'Projects outcomes and flags risk early.' },
  { title: 'Resource Optimization Engine', copy: 'Minimizes water, fertilizer, and energy waste.' },
]

const outcomes = [
  { title: 'Better Yields', copy: 'Early warnings and precise tasks prevent avoidable loss.' },
  { title: 'Lower Input Costs', copy: 'Sensors and AI dial in the exact inputs required.' },
  { title: 'Less Pesticide Waste', copy: 'Hotspot spraying targets only infected zones.' },
  { title: 'Higher ROI', copy: 'Every decision optimizes cost and profit in one loop.' },
]

const workflow = [
  { title: 'Assess', copy: 'Farmer shares quick field info while soil & climate auto-sync.' },
  { title: 'Analyze', copy: 'AI engines fuse soil, weather, drone, and rover data streams.' },
  { title: 'Act', copy: 'Get prioritized tasks, auto irrigation cues, and drone paths.' },
  { title: 'Improve', copy: 'System learns from every season to refine the next.' },
]

const hardware = [
  { title: 'Drone', copy: 'Aerial scanning + precision spraying for 35% less pesticide use.' },
  { title: 'Rover', copy: 'Row-level disease detection and micro-hotspot intel.' },
  { title: 'Soil IoT Station', copy: 'Moisture, pH, EC, climate, and irrigation automation.' },
]

const dashboardCards = [
  {
    title: "Today’s Tasks",
    lines: ['Irrigate Block B for 22 minutes', 'Drone spray hotspot 3 (2 ha)', 'Apply micronutrients to Zone 5'],
  },
  { title: 'Field Heatmap', lines: ['Stress rising at south boundary', 'Healthy canopy elsewhere', 'Pest risk: 18%'] },
  { title: 'Soil Readings', lines: ['Moisture 28% · EC 1.2 · pH 6.4 · NPK score 82%'] },
  { title: 'Yield Forecast', lines: ['Projected 4.2× ROI if current plan holds.'] },
]

const pricing = [
  {
    id: 'small',
    title: 'Small Farmers',
    price: '₹3,000',
    suffix: '/season',
    description: '1–2 hectares',
    items: ['AI insights', 'Crop recommendations', 'Daily smart tasks'],
  },
  {
    id: 'medium',
    title: 'Medium Farmers',
    price: '₹15,000',
    suffix: '/season',
    description: '2–7 hectares',
    items: ['AI + IoT bundle', 'Optional rover add-on', 'Agronomist chat'],
  },
  {
    id: 'enterprise',
    title: 'Large / FPO / Institutional',
    price: 'Custom',
    description: '7+ hectares',
    items: ['Full AI suite', 'Drone + rover fleet', 'Deployment & support'],
  },
]

const QUESTIONNAIRE_API_BASE = (import.meta.env?.VITE_QUESTIONNAIRE_API || '').trim() || 'http://localhost:7001'

type PricingTier = (typeof pricing)[number]

type AssessmentAnswers = {
  acreage: number
  soil: string
  irrigation: string
  budget: string
  goal: string
  soilType: string
  lastCrop: string
  residueLeft: string
  plantationPlan: string
  region: string
  notes: string
}

// Intercropping predictions can arrive as an array or nested under raw_results/simplified_results
type IntercroppingPrediction =
  | Record<string, unknown>
  | Record<string, unknown>[]
  | {
      raw_results?: Record<string, unknown>[]
      simplified_results?: Record<string, unknown>[]
      [key: string]: unknown
    }
  | null

type StoredUser = {
  id: string
  username: string
  email: string
  password: string
  provider: 'password' | 'google'
  createdAt: string
  avatarUrl?: string
}

type UserProfile = {
  id: string
  name: string
  avatar: string
}

type UserContextValue = {
  user: UserProfile | null
  login: () => UserProfile | null
  logout: () => void
  setUserFromAuth: (profile: UserProfile) => void
}

type PurchaseMap = Record<string, string[]>

const PURCHASED_STORAGE_KEY = 'avecAgroPurchasedPlans'
const USERS_STORAGE_KEY = 'avecAgroUsers'
const GOOGLE_CLIENT_ID = '1098550694042-s50064nq9c1sgo465rnf2gq889dghneq.apps.googleusercontent.com'
const USERS_API_ENDPOINT = import.meta.env?.VITE_USERS_API ?? 'http://localhost:6002/api/users'

declare global {
  interface Window {
    google?: {
      accounts: {
        id: {
          initialize: (options: { client_id: string; callback: (response: { credential: string }) => void }) => void
          renderButton: (element: HTMLElement | null, options: Record<string, unknown>) => void
          prompt: () => void
        }
      }
    }
  }
}

type PurchaseContextValue = {
  purchasedPlans: string[]
  markPlanPurchased: (planId: string) => void
}

const PurchaseContext = createContext<PurchaseContextValue | null>(null)
const UserContext = createContext<UserContextValue | null>(null)

function getStoredPurchaseMap(): PurchaseMap {
  if (typeof window === 'undefined') return {}
  try {
    const stored = window.localStorage.getItem(PURCHASED_STORAGE_KEY)
    return stored ? (JSON.parse(stored) as PurchaseMap) : {}
  } catch {
    return {}
  }
}

function usePurchase() {
  const context = useContext(PurchaseContext)
  if (!context) {
    throw new Error('usePurchase must be used inside PurchaseContext')
  }
  return context
}

function useUser() {
  const context = useContext(UserContext)
  if (!context) {
    throw new Error('useUser must be used inside UserContext')
  }
  return context
}

function slugify(value: string) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

function avatarForName(name: string) {
  const id = slugify(name) || 'farmer'
  return `https://i.pravatar.cc/64?u=${encodeURIComponent(id)}`
}

const seededUsers = (seedUsers as StoredUser[]) ?? []

function getStoredUsers(): StoredUser[] {
  if (typeof window === 'undefined') return seededUsers
  try {
    const raw = window.localStorage.getItem(USERS_STORAGE_KEY)
    if (raw) {
      return JSON.parse(raw) as StoredUser[]
    }
  } catch (error) {
    console.warn('Failed to parse stored users', error)
  }
  return seededUsers
}

async function persistUsersToServer(users: StoredUser[]) {
  if (typeof window === 'undefined') return
  try {
    const response = await fetch(USERS_API_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(users),
    })
    if (!response.ok) {
      const text = await response.text()
      throw new Error(text || 'Failed to persist users file')
    }
  } catch (error) {
    console.error('Unable to persist users.json via backend', error)
  }
}

function loadGoogleIdentityScript(): Promise<void> {
  if (typeof window === 'undefined') return Promise.resolve()
  if (document.getElementById('google-identity-script')) {
    return Promise.resolve()
  }
  return new Promise((resolve, reject) => {
    const script = document.createElement('script')
    script.src = 'https://accounts.google.com/gsi/client'
    script.async = true
    script.defer = true
    script.id = 'google-identity-script'
    script.onload = () => resolve()
    script.onerror = () => reject(new Error('Failed to load Google Identity Services script'))
    document.body.appendChild(script)
  })
}

const stats = [
  { title: '82% NPK Prediction Accuracy', copy: 'Benchmark-verified nutrient mapping.' },
  { title: '35% Pesticide Savings', copy: 'Hotspot spraying avoids overspray.' },
  { title: '22% Water Savings', copy: 'Automated irrigation only when required.' },
  { title: '1.8× – 4.2× ROI', copy: 'Documented across estates and FPO pilots.' },
]

const testimonials = [
  { quote: '“Avec Agro helped us cut pesticide use almost in half within two seasons.”', author: 'Farmer, Punjab' },
  { quote: '“The drone + AI workflow is a game changer for our FPO planning.”', author: 'FPO Lead, Maharashtra' },
  { quote: '“Soil IoT with smart tasks gave clarity for every irrigation cycle.”', author: 'Farmer, Karnataka' },
]

const FooterLink = ({ label }: { label: string }) => (
  <a className="text-sm text-gray-400 transition hover:text-white" href="#">
    {label}
  </a>
)

function Section({ title, copy, children }: { title: string; copy: string; children: ReactNode }) {
  return (
    <section className="section-shell space-y-10">
      <div className="space-y-4 text-center lg:text-left">
        <p className="text-sm uppercase tracking-[0.35em] text-gray-400">Avec Agro</p>
        <h2 className="section-title">{title}</h2>
        <p className="section-copy">{copy}</p>
      </div>
      {children}
    </section>
  )
}

const defaultPlan = pricing[0]

const defaultAssessmentAnswers: AssessmentAnswers = {
  acreage: 3,
  soil: 'Loamy',
  irrigation: 'Drip / Micro',
  budget: 'Balanced',
  goal: 'Higher ROI',
  soilType: '',
  lastCrop: '',
  residueLeft: '',
  plantationPlan: 'Multi height',
  region: '',
  notes: ''
}

function buildRecommendations(
  answers: AssessmentAnswers,
  predictions?: {
    npkPrediction?: Record<string, unknown> | null
    intercroppingPrediction?: IntercroppingPrediction
    multiheightPrediction?: Record<string, unknown> | null
  }
): string[] {
  const output = new Set<string>()

  // Handle Multiheight predictions (array format with Crops and Yield_Impact_%)
  if (answers.plantationPlan === 'Multi height') {
    const multiheightList = predictions?.multiheightPrediction as Array<{ Crops?: string[]; 'Yield_Impact_%'?: number }> | undefined
    if (Array.isArray(multiheightList) && multiheightList.length > 0) {
      multiheightList.forEach((item) => {
        if (Array.isArray(item.Crops) && item.Crops.length > 0) {
          // Format: User input + crop1 + crop2 (+ crop3 etc)
          const crops = item.Crops.join(' + ')
          output.add(`${answers.lastCrop} + ${crops}`)
        }
      })
    }
    // If no multiheight predictions, don't add fallback - let it show no recommendations
    if (output.size === 0) {
      console.log('[buildRecommendations] Multi height selected but no valid multiheight predictions')
      return []
    }
    return Array.from(output)
  }

  // Handle Intercropping predictions (array format with Crops and Yield_Impact_%)
  if (answers.plantationPlan === 'Intercropping') {
    const rawInter = predictions?.intercroppingPrediction as
      | Array<{ Crops?: string[]; 'Yield_Impact_%'?: number }>
      | { raw_results?: Array<{ Crops?: string[]; 'Yield_Impact_%'?: number }>; simplified_results?: Array<{ Crops?: string[]; 'Yield_Impact_%'?: number }>; [key: string]: unknown }
      | undefined

    const intercroppingList = Array.isArray(rawInter)
      ? rawInter
      : Array.isArray(rawInter?.raw_results)
        ? rawInter?.raw_results
        : Array.isArray(rawInter?.simplified_results)
          ? rawInter?.simplified_results
          : undefined

    console.log('[buildRecommendations] Intercropping - Raw prediction:', predictions?.intercroppingPrediction)
    console.log('[buildRecommendations] Intercropping - Parsed list:', intercroppingList)
    
    if (Array.isArray(intercroppingList) && intercroppingList.length > 0) {
      intercroppingList.forEach((item, idx) => {
        console.log(`[buildRecommendations] Intercropping item ${idx}:`, item)
        if (Array.isArray(item.Crops) && item.Crops.length >= 2) {
          // Format: First crop (user input) + Second crop (model prediction) (yield impact %)
          const userInputCrop = item.Crops[0]
          const modelPredictedCrop = item.Crops[1]
          const yieldImpact = item['Yield_Impact_%'] ? Number(item['Yield_Impact_%']) : null
          const yieldStr = yieldImpact ? `(${yieldImpact.toFixed(2)}% yield impact)` : ''
          const recommendation = `${userInputCrop} + ${modelPredictedCrop} ${yieldStr}`.trim()
          console.log(`[buildRecommendations] Adding recommendation: ${recommendation}`)
          output.add(recommendation)
        }
      })
    }
    
    // If no intercropping predictions, don't add fallback - let it show no recommendations
    if (output.size === 0) {
      console.log('[buildRecommendations] Intercropping selected but no valid intercropping predictions')
      return []
    }
    console.log('[buildRecommendations] Intercropping final output:', Array.from(output))
    return Array.from(output)
  }

  // For Single crop or other plans, use NPK prediction if available
  const modelSuggestedCrop =
    predictions?.npkPrediction?.recommended_crop ||
    predictions?.npkPrediction?.predicted_crop ||
    predictions?.npkPrediction?.crop ||
    null

  if (modelSuggestedCrop && String(modelSuggestedCrop).trim().length > 0) {
    output.add(`${String(modelSuggestedCrop)} (AI recommendation for your field)`)
  }

  const acreage = Number.isFinite(answers.acreage) ? answers.acreage : 0

  if (acreage > 0 && acreage < 2) {
    output.add('High-density Strawberry tunnels')
    output.add('Leafy greens with staggered harvests')
  } else if (acreage >= 5 && acreage <= 10) {
    output.add('Hybrid Maize + Turmeric rotation')
  } else if (acreage > 10) {
    output.add('Paddy ↔ Mustard dual cropping at estate blocks')
  } else {
    output.add('Paddy ↔ Mustard dual cropping')
  }

  if (answers.soil === 'Sandy Loam') {
    output.add('Groundnut with integrated drip fertigation')
  } else if (answers.soil === 'Clay / Heavy') {
    output.add('Short-duration pulses before Kharif paddy')
  } else {
    output.add('Baby corn + vegetable intercrop pack')
  }

  if (answers.budget === 'Lean') {
    output.add('Bio-fortified Millets (low input)')
  } else if (answers.budget === 'Aggressive') {
    output.add('Greenhouse Cucumbers for premium markets')
  }

  if (answers.goal === 'Higher ROI') {
    output.add('Onion + Chilli stagger for cashflow')
  } else if (answers.goal === 'Risk diversification') {
    output.add('Oilseed + pulse pairing to hedge price risk')
  } else {
    output.add('Fodder maize plus dairy integration')
  }

  if (answers.irrigation.includes('Rainfed')) {
    output.add('Short-duration Sorghum with resilient hybrids')
  } else {
    output.add('Precision horticulture block with fertigation')
  }

  const lastCrop = answers.lastCrop.trim().toLowerCase()
  if (lastCrop.includes('paddy') || lastCrop.includes('rice')) {
    output.add('Include a moong or urad break after paddy to rebuild soil nitrogen')
  } else if (lastCrop.includes('cotton') || lastCrop.includes('sugarcane')) {
    output.add('Short-duration pulses to reset bollworm and borer cycles')
  }

  const residue = answers.residueLeft.trim().toLowerCase()
  if (residue === 'yes') {
    output.add('Residue mulching + microbial consortia for faster breakdown')
  } else if (residue === 'no') {
    output.add('Fast establishing cover crop strip to rebuild organic carbon')
  }

  const soilType = answers.soilType.trim().toLowerCase()
  if (soilType.includes('black') || soilType.includes('vertisol')) {
    output.add('Deep-rooted pigeon pea rows to loosen tight black soils')
  } else if (soilType.includes('saline') || soilType.includes('alkaline')) {
    output.add('Salt-tolerant barley + gypsum-based soil rebalancing')
  }

  if (answers.plantationPlan === 'Multi height') {
    output.add('Banana + ginger + fodder grass multi-tier block')
  } else if (answers.plantationPlan === 'Intercropping') {
    output.add('Marigold strips for natural pest breaks inside intercrops')
  } else if (answers.plantationPlan === 'Single crop') {
    output.add('High-density single crop with drone scouting for uniform stands')
  }

  if (answers.notes.trim().length > 0) {
    output.add('Schedule an agronomist consult to factor in custom constraints')
  }

  return Array.from(output)
}

function usePlanSelection() {
  const location = useLocation()
  const possiblePlan = (location.state as { plan?: PricingTier } | undefined)?.plan
  return possiblePlan ?? defaultPlan
}

function LandingPage() {
  const { user, login, logout } = useUser()
  // Get fetch functions from weather logic hook with plantationType parameter
  const {
    fetchWeatherData,
    fetchNpkAndIntercroppingData,
    fetchMultiheightData,
    npkPrediction,
    intercroppingPrediction,
    multiheightPrediction,
  } = useWeatherLogic()
  const navigate = useNavigate()
  const { purchasedPlans } = usePurchase()
  const [assessmentOpen, setAssessmentOpen] = useState(false)
  const [assessmentStep, setAssessmentStep] = useState<'questions' | 'recommendations'>('questions')
  const [assessmentAnswers, setAssessmentAnswers] = useState<AssessmentAnswers>(defaultAssessmentAnswers)
  const [assessmentRecommendations, setAssessmentRecommendations] = useState<string[]>([])
  const [isLoadingPredictions, setIsLoadingPredictions] = useState(false)
  const [demoOpen, setDemoOpen] = useState(false)
  const [selectedCrop, setSelectedCrop] = useState<string | null>(null)
  const [questionnaireStatus, setQuestionnaireStatus] = useState<'idle' | 'saving'>('idle')
  const [questionnaireError, setQuestionnaireError] = useState<string | null>(null)
  const [questionnaireEntryId, setQuestionnaireEntryId] = useState<number | null>(null)
  const [cropConfirmed, setCropConfirmed] = useState(false)
  const hasActivePlan = purchasedPlans.length > 0
  const hasCompletedAssessment = assessmentRecommendations.length > 0
  const isSavingQuestionnaire = questionnaireStatus === 'saving'

  // Listen for external generateRecommendation events and trigger predictions based on plantationType
  useEffect(() => {
    const handleGenerate = () => {
      setIsLoadingPredictions(true)
      const plantationType = assessmentAnswers.plantationPlan
      console.log('🌾 [LandingPage] generateRecommendation event received with plantationType:', plantationType)
      console.log('🌾 [LandingPage] Prediction logic: NPK always, intercropping if "Intercropping", multiheight if "Multi height"')
      
      // NPK is always called
      if (typeof fetchWeatherData === 'function') {
        console.log('🌾 [LandingPage] ▶️ Calling fetchWeatherData (NPK - always)')
        void fetchWeatherData()
      }
      
      // Intercropping only if plantationType is "Intercropping"
      if (plantationType === 'Intercropping' && typeof fetchNpkAndIntercroppingData === 'function') {
        console.log('🌾 [LandingPage] ▶️ Calling fetchNpkAndIntercroppingData (because plantationType is Intercropping)')
        void fetchNpkAndIntercroppingData()
      }
      
      // Multiheight only if plantationType is "Multi height"
      if (plantationType === 'Multi height' && typeof fetchMultiheightData === 'function') {
        console.log('🌾 [LandingPage] ▶️ Calling fetchMultiheightData (because plantationType is Multi height)')
        void fetchMultiheightData()
      }
      
      if (plantationType !== 'Intercropping' && plantationType !== 'Multi height') {
        console.log('🌾 [LandingPage] ℹ️ No intercropping or multiheight prediction for plantationType:', plantationType)
      }
    }
    window.addEventListener('generateRecommendation', handleGenerate)
    return () => window.removeEventListener('generateRecommendation', handleGenerate)
  }, [fetchWeatherData, fetchNpkAndIntercroppingData, fetchMultiheightData, assessmentAnswers.plantationPlan])

  // Rebuild recommendations whenever predictions update
  useEffect(() => {
    if (assessmentStep === 'recommendations' && hasCompletedAssessment) {
      console.log('🔄 [LandingPage] Predictions updated, rebuilding recommendations...')
      const updatedRecs = buildRecommendations(assessmentAnswers, { npkPrediction, intercroppingPrediction, multiheightPrediction })
      console.log('🔄 [LandingPage] Updated recommendations:', updatedRecs)
      setAssessmentRecommendations(updatedRecs)
      // Turn off loading once we have recommendations
      if (updatedRecs.length > 0) {
        setIsLoadingPredictions(false)
      }
    }
  }, [npkPrediction, intercroppingPrediction, multiheightPrediction, assessmentAnswers, assessmentStep, hasCompletedAssessment])

  const handleChoosePlan = (tier: PricingTier) => {
    const activeUser = user ?? login()
    if (!activeUser) return
    navigate('/confirm', { state: { plan: tier } })
  }

  const openAssessment = () => {
    if (!user) {
      const newUser = login()
      if (!newUser) return
    }
    setAssessmentAnswers(defaultAssessmentAnswers)
    setAssessmentRecommendations([])
    setAssessmentStep('questions')
    setAssessmentOpen(true)
    setSelectedCrop(null)
    setCropConfirmed(false)
    setQuestionnaireEntryId(null)
  }

  const closeAssessment = () => {
    setAssessmentOpen(false)
  }

  const handleAssessmentSubmit = async () => {
    console.log('📋 [handleAssessmentSubmit] Form submitted, building recommendations...')
    const recs = buildRecommendations(assessmentAnswers, { npkPrediction, intercroppingPrediction, multiheightPrediction })
    console.log('📋 [handleAssessmentSubmit] Recommendations built:', recs)
    setAssessmentRecommendations(recs)
    setAssessmentStep('recommendations')
    setSelectedCrop(null)
    setQuestionnaireError(null)
    setCropConfirmed(false)

    if (!user) {
      alert('Please log in to save your questionnaire.')
      navigate('/auth')
      return
    }

    try {
      setQuestionnaireStatus('saving')
      const payload = {
        username: user.name,
        planId: purchasedPlans[0] ?? null,
        answers: assessmentAnswers,
        selectedCrop: null,
        recommendations: recs,
      }
      console.log('📋 [handleAssessmentSubmit] Posting questionnaire to backend:', payload)
      const response = await fetch(`${QUESTIONNAIRE_API_BASE}/api/questionnaire`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || 'Failed to save questionnaire')
      }
      const data = await response.json()
      console.log('📋 [handleAssessmentSubmit] ✅ Questionnaire saved with entry ID:', data?.entry?.id)
      setQuestionnaireEntryId(data?.entry?.id ?? null)
      
      console.log('✅ [handleAssessmentSubmit] Dispatching generateRecommendation event now...')
      window.dispatchEvent(new Event('generateRecommendation'))
      console.log('✅ [handleAssessmentSubmit] Event dispatched! Predictions should start firing...')
    } catch (error) {
      console.error('❌ [handleAssessmentSubmit] Error:', error)
      setQuestionnaireError('Unable to save questionnaire details. Please try again.')
    } finally {
      setQuestionnaireStatus('idle')
    }
  }

  const openDemo = () => setDemoOpen(true)
  const closeDemo = () => setDemoOpen(false)
  const handleSelectCrop = (crop: string) => {
    setSelectedCrop(crop)
    setQuestionnaireError(null)
    setCropConfirmed(false)
  }

  const confirmCropSelection = async () => {
    if (!selectedCrop) {
      setQuestionnaireError('Select a crop before confirming your choice.')
      return
    }

    if (!questionnaireEntryId) {
      setQuestionnaireError('Questionnaire not saved yet. Generate recommendations again.')
      return
    }

    try {
      setQuestionnaireStatus('saving')
      const response = await fetch(`${QUESTIONNAIRE_API_BASE}/api/questionnaire/${questionnaireEntryId}/selection`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          selectedCrop,
          recommendations: assessmentRecommendations,
        }),
      })

      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || 'Failed to save crop selection')
      }

      setCropConfirmed(true)
      setQuestionnaireError(null)
    } catch (error) {
      console.error('Unable to save crop selection', error)
      setQuestionnaireError('Unable to save crop selection. Please try again.')
    } finally {
      setQuestionnaireStatus('idle')
    }
  }

  const scrollToPricing = () => {
    if (typeof document === 'undefined') return
    const section = document.getElementById('pricing')
    section?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  const handleStartJourney = async () => {
    if (!user) {
      alert('Please log in to continue. You will be redirected to the auth page.')
      navigate('/auth')
      return
    }

    if (!hasActivePlan) {
      alert('Activate a plan before accessing the dashboard.')
      scrollToPricing()
      return
    }

    if (!hasCompletedAssessment) {
      alert('Complete the soil & crop questionnaire first.')
      openAssessment()
      return
    }

    if (!selectedCrop) {
      alert('Select one of the recommended crops to proceed.')
      setAssessmentOpen(true)

      setAssessmentStep('recommendations')
      return
    }

    if (!cropConfirmed) {
      alert('Confirm your crop selection before continuing.')
      setAssessmentOpen(true)
      setAssessmentStep('recommendations')
      return
    }

    window.parent?.postMessage({
      type: 'agri-lp-continue',
      payload: { target: 'dashboard', username: user.name },
    }, '*')
  }

  return (
    <>
    <div className="bg-gray-900 text-white">
      <header className="relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(52,211,153,0.25),_transparent_55%)]" />
        <div className="section-shell relative py-10 lg:py-16">
          <nav className="glass-card mb-12 flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <img src={logoImage} alt="Avec Agro logo" className="h-10 w-10 rounded-full border border-white/20 object-cover" />
              <span className="font-display text-sm tracking-[0.6em] text-gray-200">AVEC AGRO</span>
            </div>
            <div className="flex flex-wrap items-center gap-3 text-sm font-semibold text-gray-300">
              <a className="hover:text-white" href="#pricing">
                Pricing
              </a>
              <a className="hover:text-white" href="#testimonials">
                Stories
              </a>
              <a className="hover:text-white" href="#contact">
                Contact
              </a>
              <button className="button-ghost" onClick={openDemo}>Watch Demo</button>
              <button className="button-solid" onClick={openAssessment}>
                Start Free Assessment
              </button>
              {user ? (
                <div className="flex items-center gap-3 rounded-full border border-white/20 bg-white/5 px-3 py-1.5 text-white">
                  <img
                    src={user.avatar}
                    alt={user.name}
                    className="h-9 w-9 rounded-full border border-white/30 object-cover"
                  />
                  <span className="text-sm font-semibold">{user.name}</span>
                  <button onClick={logout} className="text-xs text-gray-400 hover:text-white">
                    Logout
                  </button>
                </div>
              ) : (
                <button onClick={() => navigate('/auth')} className="button-ghost">
                  Log in
                </button>
              )}
            </div>
          </nav>

          <div className="grid gap-10 lg:grid-cols-[1.1fr_0.9fr] lg:items-center">
            <div className="space-y-8">
              <p className="text-sm uppercase tracking-[0.4em] text-gray-400">Autonomous Farm Intelligence</p>
              <div className="space-y-4">
                <h1 className="text-4xl font-semibold leading-tight md:text-5xl">
                  <span className="gradient-text">Autonomous farming</span> for the real world.
                </h1>
                <p className="text-lg text-gray-300">
                  AI-powered insights, drone automation, and intelligent field monitoring—engineered for Indian agriculture.
                </p>
              </div>
              <div className="flex flex-wrap gap-3">
                <button className="button-solid" onClick={openAssessment}>
                  Start Free Assessment
                </button>
                <button className="button-ghost" onClick={openDemo}>Watch Demo</button>
              </div>
              <p className="text-sm text-gray-400">Trusted by farmers, validated by scientists, engineered for India.</p>
            </div>

            <div className="glass-card space-y-6">
              <div className="rounded-2xl bg-white/5 p-5">
                <p className="text-sm text-gray-400">Live ecosystem graph</p>
                <div className="mt-4 grid grid-cols-3 gap-4 text-center text-sm">
                  {heroStats.map((stat) => (
                    <div key={stat.label}>
                      <p className="text-2xl font-semibold text-white">{stat.value}</p>
                      <p className="text-gray-400">{stat.label}</p>
                    </div>
                  ))}
                </div>
              </div>
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="chart-container">
                  <p className="text-sm text-gray-400">Drone Coverage</p>
                  <p className="mt-4 text-3xl font-semibold">70 acres / day</p>
                  <p className="text-sm text-gray-400">Precision spraying with hotspot targeting.</p>
                </div>
                <div className="chart-container">
                  <p className="text-sm text-gray-400">Rover Insights</p>
                  <p className="mt-4 text-3xl font-semibold">11 active alerts</p>
                  <p className="text-sm text-gray-400">Leaf stress, nutrient gaps, pest vectors.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      <section className="border-y border-white/10 bg-gray-900/60">
        <div className="section-shell flex flex-col gap-4 py-6 text-center text-sm font-semibold text-gray-400 lg:flex-row lg:items-center lg:justify-between">
          <span>Backed across India</span>
          <div className="flex flex-wrap justify-center gap-6 text-gray-500">
            {trustLogos.map((logo) => (
              <span key={logo}>{logo}</span>
            ))}
          </div>
        </div>
      </section>

      <main className="space-y-20 py-16">
        <Section
          title="Built to simplify decision-making on every Indian farm."
          copy="Clear guidance, daily actions, and continuous monitoring wrapped into a single intelligence loop."
        >
          <div className="grid gap-6 md:grid-cols-3">
            {features.map((feature) => (
              <article key={feature.title} className="glass-card hover-lift space-y-3">
                <h3 className="text-xl font-semibold">{feature.title}</h3>
                <p className="text-muted">{feature.description}</p>
              </article>
            ))}
          </div>
        </Section>

        <Section
          title="The 8-engine autonomous intelligence platform"
          copy="Every engine functions like an organ inside one living ecosystem, constantly learning from field reality."
        >
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {engines.map((engine) => (
              <article key={engine.title} className="glass-card hover-lift space-y-2">
                <h3 className="text-lg font-semibold">{engine.title}</h3>
                <p className="text-muted text-sm">{engine.copy}</p>
              </article>
            ))}
          </div>
        </Section>

        <Section title="Real outcomes farmers care about" copy="Measurable impact across yield, cost, and peace of mind.">
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            {outcomes.map((outcome) => (
              <article key={outcome.title} className="glass-card hover-lift space-y-2">
                <h3 className="text-xl font-semibold">{outcome.title}</h3>
                <p className="text-muted">{outcome.copy}</p>
              </article>
            ))}
          </div>
        </Section>

        <Section title="A simple workflow for every farm" copy="Four practical steps keep the loop clear and actionable.">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {workflow.map((step, index) => (
              <article key={step.title} className="glass-card hover-lift space-y-3 text-center">
                <p className="text-sm uppercase tracking-[0.3em] text-gray-500">0{index + 1}</p>
                <h3 className="text-xl font-semibold">{step.title}</h3>
                <p className="text-muted">{step.copy}</p>
              </article>
            ))}
          </div>
        </Section>

        <Section title="Hardware ecosystem" copy="Purpose-built devices capture reality and feed the AI loop in real time.">
          <div className="grid gap-6 md:grid-cols-3">
            {hardware.map((item) => (
              <article key={item.title} className="glass-card hover-lift space-y-3">
                <h3 className="text-xl font-semibold">{item.title}</h3>
                <p className="text-muted">{item.copy}</p>
              </article>
            ))}
          </div>
        </Section>

        <Section title="Dashboard preview" copy="Tasks, heatmaps, soil readings, spray plans, and yield forecasts in one clean UI.">
          <div className="grid gap-6 md:grid-cols-2">
            {dashboardCards.map((card) => (
              <article key={card.title} className="glass-card hover-lift space-y-3">
                <h3 className="text-lg font-semibold">{card.title}</h3>
                <ul className="space-y-2 text-sm text-gray-300">
                  {card.lines.map((line) => (
                    <li key={line}>{line}</li>
                  ))}
                </ul>
              </article>
            ))}
          </div>
        </Section>

        <Section title="Pricing" copy="Simple, transparent, farmer-first plans that scale with acreage." >
          <div id="pricing" className="grid gap-6 md:grid-cols-3">
            {pricing.map((tier) => (
              <article key={tier.title} className="glass-card hover-lift space-y-4">
                <div>
                  <p className="text-sm uppercase tracking-[0.3em] text-gray-500">{tier.title}</p>
                  <p className="mt-2 text-4xl font-semibold text-white">
                    {tier.price}
                    {tier.suffix && <span className="text-base font-medium text-gray-400">{tier.suffix}</span>}
                  </p>
                  <p className="text-sm text-gray-400">{tier.description}</p>
                </div>
                <ul className="space-y-2 text-sm text-gray-300">
                  {tier.items.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
                {purchasedPlans.includes(tier.id) ? (
                  <span className="inline-flex w-full items-center justify-center rounded-full border border-emerald-400/60 bg-emerald-400/10 px-6 py-3 text-sm font-semibold text-emerald-300">
                    Activated
                  </span>
                ) : (
                  <button className="button-ghost w-full" onClick={() => handleChoosePlan(tier)}>
                    Choose Plan
                  </button>
                )}
              </article>
            ))}
          </div>
        </Section>

        <Section title="Validation & impact" copy="Real-world data instead of hype. Numbers evaluators trust.">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {stats.map((stat) => (
              <article key={stat.title} className="glass-card hover-lift space-y-2 text-center">
                <h3 className="text-xl font-semibold">{stat.title}</h3>
                <p className="text-muted">{stat.copy}</p>
              </article>
            ))}
          </div>
        </Section>

        <Section title="Farmers & partners" copy="Short, believable stories from the field.">
          <div id="testimonials" className="grid gap-6 md:grid-cols-3">
            {testimonials.map((testimonial) => (
              <article key={testimonial.author} className="glass-card hover-lift space-y-4">
                <p className="text-lg">{testimonial.quote}</p>
                <p className="text-sm font-semibold text-gray-400">{testimonial.author}</p>
              </article>
            ))}
          </div>
        </Section>
      </main>

      <section id="contact" className="section-shell py-16">
        <div className="glass-card bg-emerald-500/10 text-center">
          <h2 className="section-title">Ready to transform your farm?</h2>
          <p className="section-copy">Start with a free soil & crop assessment. No pressure, just clarity.</p>
          <div className="mt-6 flex flex-wrap justify-center gap-4">
            <button
              className="button-solid disabled:cursor-not-allowed disabled:opacity-60"
              onClick={handleStartJourney}
              disabled={isSavingQuestionnaire}
            >
              {isSavingQuestionnaire ? 'Preparing dashboard…' : 'Start Now'}
            </button>
            <button className="button-ghost">Contact Sales</button>
          </div>
          {!hasActivePlan && (
            <p className="mt-3 text-sm text-yellow-300">
              Activate a plan to unlock the dashboard.
            </p>
          )}
          {questionnaireError && (
            <p className="mt-3 text-sm text-rose-300">{questionnaireError}</p>
          )}
        </div>
      </section>

      <footer className="border-t border-white/10 bg-black/70">
        <div className="section-shell flex flex-col gap-6 py-10 text-sm text-gray-400 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="font-display text-sm tracking-[0.6em] text-white">AVEC AGRO</p>
            <p>Autonomous Farm Intelligence</p>
          </div>
          <div className="flex flex-wrap gap-4">
            {['About', 'Contact', 'Privacy', 'Partner with us'].map((label) => (
              <FooterLink key={label} label={label} />
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <button className="button-ghost">English</button>
            <button className="button-ghost">हिंदी</button>
            <div className="flex gap-3 text-white/70">
              <span aria-label="LinkedIn">in</span>
              <span aria-label="YouTube">▶</span>
              <span aria-label="Twitter">X</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
    {assessmentOpen && (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4 py-8">
        <div className="absolute inset-0" onClick={closeAssessment} />
        <div className="glass-card relative z-10 max-h-[90vh] w-full max-w-2xl overflow-y-auto bg-gray-900/90 p-8">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.4em] text-gray-500">Avec Agro</p>
              <h3 className="text-2xl font-semibold">
                {assessmentStep === 'questions' ? 'Soil & crop questionnaire' : 'Recommended crop rotations'}
              </h3>
            </div>
            <button className="text-gray-400 transition hover:text-white" onClick={closeAssessment}>
              ✕
            </button>
          </div>
          {assessmentStep === 'questions' ? (
            <form
              className="space-y-5"
              onSubmit={(event) => {
                event.preventDefault()
                handleAssessmentSubmit()
              }}
            >
              <label className="block space-y-2 text-sm text-gray-300">
                Land owned (acres)
                <input
                  className="w-full rounded-2xl border border-white/15 bg-black/30 px-4 py-3 text-white"
                  type="number"
                  min={0}
                  step="0.1"
                  placeholder="e.g., 3.5"
                  value={Number.isFinite(assessmentAnswers.acreage) ? assessmentAnswers.acreage : ''}
                  onChange={(event) => {
                    const raw = event.target.value
                    const parsed = raw === '' ? 0 : Number(raw)
                    setAssessmentAnswers((prev) => ({ ...prev, acreage: Number.isNaN(parsed) ? 0 : parsed }))
                  }}
                />
              </label>
              <label className="block space-y-2 text-sm text-gray-300">
                Dominant soil profile
                <select
                  className="w-full rounded-2xl border border-white/15 bg-black/30 px-4 py-3 text-white"
                  value={assessmentAnswers.soil}
                  onChange={(event) => setAssessmentAnswers((prev) => ({ ...prev, soil: event.target.value }))}
                >
                  {['Loamy', 'Sandy Loam', 'Clay / Heavy'].map((option) => (
                    <option key={option} className="bg-gray-900" value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>
              <label className="block space-y-2 text-sm text-gray-300">
                Irrigation setup
                <select
                  className="w-full rounded-2xl border border-white/15 bg-black/30 px-4 py-3 text-white"
                  value={assessmentAnswers.irrigation}
                  onChange={(event) => setAssessmentAnswers((prev) => ({ ...prev, irrigation: event.target.value }))}
                >
                  {['Drip / Micro', 'Flood / Canal', 'Rainfed + supplemental'].map((option) => (
                    <option key={option} className="bg-gray-900" value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>
              <label className="block space-y-2 text-sm text-gray-300">
                Investment appetite
                <select
                  className="w-full rounded-2xl border border-white/15 bg-black/30 px-4 py-3 text-white"
                  value={assessmentAnswers.budget}
                  onChange={(event) => setAssessmentAnswers((prev) => ({ ...prev, budget: event.target.value }))}
                >
                  {['Lean', 'Balanced', 'Aggressive'].map((option) => (
                    <option key={option} className="bg-gray-900" value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>
              <label className="block space-y-2 text-sm text-gray-300">
                Primary goal for this season
                <select
                  className="w-full rounded-2xl border border-white/15 bg-black/30 px-4 py-3 text-white"
                  value={assessmentAnswers.goal}
                  onChange={(event) => setAssessmentAnswers((prev) => ({ ...prev, goal: event.target.value }))}
                >
                  {['Higher ROI', 'Risk diversification', 'Fodder security'].map((option) => (
                    <option key={option} className="bg-gray-900" value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>
              <label className="block space-y-2 text-sm text-gray-300">
                Soil type observations (lab / field)
                <input
                  className="w-full rounded-2xl border border-white/15 bg-black/30 px-4 py-3 text-white"
                  type="text"
                  placeholder="e.g., Medium black cotton, pH 7.2"
                  value={assessmentAnswers.soilType}
                  onChange={(event) => setAssessmentAnswers((prev) => ({ ...prev, soilType: event.target.value }))}
                />
              </label>
              <label className="block space-y-2 text-sm text-gray-300">
                Last crop harvested
                <input
                  className="w-full rounded-2xl border border-white/15 bg-black/30 px-4 py-3 text-white"
                  type="text"
                  placeholder="e.g., Paddy (Kharif 2024)"
                  value={assessmentAnswers.lastCrop}
                  onChange={(event) => setAssessmentAnswers((prev) => ({ ...prev, lastCrop: event.target.value }))}
                />
              </label>
              <label className="block space-y-2 text-sm text-gray-300">
                Residue left on the field
                <select
                  className="w-full rounded-2xl border border-white/15 bg-black/30 px-4 py-3 text-white"
                  value={assessmentAnswers.residueLeft}
                  onChange={(event) => setAssessmentAnswers((prev) => ({ ...prev, residueLeft: event.target.value }))}
                >
                  <option className="bg-gray-900" value="">
                    Select an option
                  </option>
                  <option className="bg-gray-900" value="Yes">
                    Yes
                  </option>
                  <option className="bg-gray-900" value="No">
                    No
                  </option>
                </select>
              </label>
              <label className="block space-y-2 text-sm text-gray-300">
                Plantation type planned for this season
                <select
                  className="w-full rounded-2xl border border-white/15 bg-black/30 px-4 py-3 text-white"
                  value={assessmentAnswers.plantationPlan}
                  onChange={(event) => setAssessmentAnswers((prev) => ({ ...prev, plantationPlan: event.target.value }))}
                >
                  {['Multi height', 'Single crop', 'Intercropping'].map((option) => (
                    <option key={option} className="bg-gray-900" value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>
              <label className="block space-y-2 text-sm text-gray-300">
                Region / Sub-region
                <input
                  className="w-full rounded-2xl border border-white/15 bg-black/30 px-4 py-3 text-white"
                  type="text"
                  placeholder="e.g., Northern Plains, Deccan Plateau, Coastal region"
                  value={assessmentAnswers.region}
                  onChange={(event) => setAssessmentAnswers((prev) => ({ ...prev, region: event.target.value }))}
                />
              </label>
              <label className="block space-y-2 text-sm text-gray-300">
                Anything else that would help our agronomists
                <textarea
                  className="w-full rounded-2xl border border-white/15 bg-black/30 px-4 py-3 text-white"
                  rows={3}
                  placeholder="Constraints, pest alerts, labor availability, nearby market demand, etc."
                  value={assessmentAnswers.notes}
                  onChange={(event) => setAssessmentAnswers((prev) => ({ ...prev, notes: event.target.value }))}
                />
              </label>
              <div className="flex flex-wrap gap-3">
                <button
                  className="button-solid"
                  type="submit"
                  onClick={() => {
                    // Trigger predictions immediately on click (also submits the form)
                    setIsLoadingPredictions(true)
                    const planType = assessmentAnswers.plantationPlan

                    if (typeof fetchWeatherData === 'function') {
                      void fetchWeatherData()
                    }

                    if (planType === 'Intercropping' && typeof fetchNpkAndIntercroppingData === 'function') {
                      void fetchNpkAndIntercroppingData()
                    }

                    if (planType === 'Multi height' && typeof fetchMultiheightData === 'function') {
                      void fetchMultiheightData()
                    }
                  }}
                >
                  Generate recommendations
                </button>
                <button className="button-ghost" onClick={closeAssessment} type="button">
                  Cancel
                </button>
              </div>
            </form>
          ) : (
            <div className="space-y-6">
              <div>
                <h4 className="text-lg font-semibold text-white mb-4">💡 Recommended Crop Rotations</h4>
                <p className="text-sm text-gray-400 mb-2">
                  Tailored suggestions based on your acreage, soil, irrigation, and budget inputs.
                </p>
              </div>

              <div className="space-y-3">
                {isLoadingPredictions ? (
                  <div className="space-y-2">
                    <div className="rounded-2xl border border-emerald-400/30 bg-emerald-400/5 p-4">
                      <div className="flex items-center gap-3">
                        <div className="flex gap-1">
                          <div className="h-2 w-2 rounded-full bg-emerald-400 animate-bounce" style={{ animationDelay: '0s' }} />
                          <div className="h-2 w-2 rounded-full bg-emerald-400 animate-bounce" style={{ animationDelay: '0.2s' }} />
                          <div className="h-2 w-2 rounded-full bg-emerald-400 animate-bounce" style={{ animationDelay: '0.4s' }} />
                        </div>
                        <span className="text-sm text-emerald-300 font-medium">Analyzing soil & climate data...</span>
                      </div>
                    </div>
                    <p className="text-xs text-gray-500 text-center mt-2">This may take a few seconds</p>
                  </div>
                ) : assessmentRecommendations.length > 0 ? (
                  assessmentRecommendations.map((item) => {
                    const active = selectedCrop === item
                    return (
                      <button
                        key={item}
                        type="button"
                        onClick={() => handleSelectCrop(item)}
                        className={`w-full rounded-2xl border px-4 py-3 text-left text-sm transition ${
                          active
                            ? 'border-emerald-400 bg-emerald-400/10 text-white'
                            : 'border-white/10 bg-white/5 text-gray-200 hover:border-emerald-300/60'
                        }`}
                      >
                        <div className="flex items-center justify-between gap-4">
                          <span>{item}</span>
                          {active && <span className="text-xs text-emerald-300">Selected</span>}
                        </div>
                      </button>
                    )
                  })
                ) : (
                  <div className="rounded-2xl border border-white/15 bg-white/5 p-4 text-sm text-gray-400">
                    No recommendations available yet. Try generating again.
                  </div>
                )}
              </div>

              {questionnaireError && (
                <p className="text-xs text-rose-300">{questionnaireError}</p>
              )}
              {selectedCrop && (
                <div className="rounded-2xl border border-emerald-300/40 bg-emerald-300/5 px-4 py-3 text-sm">
                  <div className="flex flex-wrap items-center justify-between gap-4">
                    <div>
                      <p className="text-xs uppercase tracking-wide text-emerald-200/80">Selected crop</p>
                      <p className="text-base font-semibold text-white">{selectedCrop}</p>
                    </div>
                    <button
                      type="button"
                      onClick={confirmCropSelection}
                      disabled={cropConfirmed || isSavingQuestionnaire}
                      className={`rounded-full px-4 py-2 text-xs font-semibold transition ${
                        cropConfirmed
                          ? 'bg-emerald-400/10 text-emerald-200'
                          : isSavingQuestionnaire
                            ? 'bg-emerald-400/20 text-emerald-100 cursor-not-allowed'
                            : 'bg-emerald-400 text-emerald-950 hover:bg-emerald-300'
                      }`}
                    >
                      {cropConfirmed ? 'Crop Confirmed' : isSavingQuestionnaire ? 'Saving…' : 'Confirm Selection'}
                    </button>
                  </div>
                  <p className="mt-2 text-xs text-emerald-200/70">
                    {cropConfirmed
                      ? 'Ready to redirect—finish and head to your dashboard.'
                      : 'Confirm this crop to save the questionnaire and enable dashboard redirect.'}
                  </p>
                  {cropConfirmed && (
                    <button
                      type="button"
                      onClick={handleStartJourney}
                      disabled={isSavingQuestionnaire}
                      className={`mt-4 w-full rounded-full px-4 py-2 text-sm font-semibold transition ${
                        isSavingQuestionnaire
                          ? 'bg-emerald-400/20 text-emerald-100 cursor-not-allowed'
                          : 'bg-emerald-400 text-emerald-950 hover:bg-emerald-300'
                      }`}
                    >
                      {isSavingQuestionnaire ? 'Redirecting…' : 'Redirect to Dashboard'}
                    </button>
                  )}
                </div>
              )}
              <div className="flex flex-wrap gap-3">
                <button
                  className="button-solid"
                  onClick={() => {
                    setAssessmentStep('questions')
                  }}
                >
                  Try different inputs
                </button>
                <button className="button-ghost" onClick={closeAssessment}>
                  Close
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    )}
    {demoOpen && (
      <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/70 px-4 py-8">
        <div className="absolute inset-0" onClick={closeDemo} />
        <div className="glass-card relative z-10 w-full max-w-4xl bg-gray-900/90 p-6">
          <div className="mb-4 flex items-center justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.4em] text-gray-500">Demo</p>
              <h3 className="text-2xl font-semibold text-white">Autonomous farm command center</h3>
            </div>
            <button className="text-gray-400 transition hover:text-white" onClick={closeDemo}>
              ✕
            </button>
          </div>
          <div className="aspect-video w-full overflow-hidden rounded-2xl border border-white/10 bg-black">
              <iframe
              className="h-full w-full"
              src="https://www.youtube.com/embed/4V2clnjDmFo?si=LAU9BHuaLdK5vb2l"
              title="Avec Agro Demo"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            />
          </div>
          <p className="mt-4 text-sm text-gray-400">
            This short walkthrough highlights drone orchestration, rover alerts, and agronomist collaboration tools.
          </p>
        </div>
      </div>
    )}
    </>
  )
}

function ConfirmationPage() {
  const navigate = useNavigate()
  const plan = usePlanSelection()
  const { user, login } = useUser()
  const { purchasedPlans } = usePurchase()
  const isActivated = purchasedPlans.includes(plan.id)

  if (!user) {
    return (
      <div className="min-h-screen bg-gray-950 text-white">
        <div className="section-shell space-y-6 py-16 text-center">
          <h1 className="text-3xl font-semibold">Log in to continue</h1>
          <p className="section-copy">Please log in to confirm your plan selection.</p>
          <div className="flex flex-wrap justify-center gap-4">
            <button className="button-solid" onClick={login}>
              Log in
            </button>
            <button className="button-ghost" onClick={() => navigate('/')}>Back to landing</button>
          </div>
        </div>
      </div>
    )
  }
  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <div className="section-shell space-y-8 py-16">
        <div className="flex items-center justify-between gap-4">
          <button className="text-sm text-gray-400 transition hover:text-white" onClick={() => navigate('/')}>
            ← Back to landing
          </button>
          <p className="text-xs uppercase tracking-[0.4em] text-gray-500">Step 1 · Confirm</p>
        </div>
        <div className="space-y-4 text-center">
          <h1 className="text-4xl font-semibold">Confirm your {plan.title} plan</h1>
          <p className="section-copy">Review inclusions before heading to secure payment.</p>
        </div>
        <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
          <article className="glass-card space-y-4">
            <div>
              <p className="text-sm uppercase tracking-[0.25em] text-gray-500">Selected plan</p>
              <p className="mt-3 text-3xl font-semibold text-white">
                {plan.price}
                {plan.suffix && <span className="text-base font-medium text-gray-400">{plan.suffix}</span>}
              </p>
              <p className="text-sm text-gray-400">{plan.description}</p>
            </div>
            <ul className="space-y-2 text-sm text-gray-300">
              {plan.items.map((item) => (
                <li key={item} className="flex items-center gap-2">
                  <span className="text-emerald-400">•</span>
                  {item}
                </li>
              ))}
            </ul>
          </article>
          <article className="glass-card space-y-3">
            <p className="text-sm text-gray-400">Deployment preferences</p>
            <ul className="space-y-2 text-sm text-gray-300">
              <li>• Onboarding call within 24 hours</li>
              <li>• Soil + crop data sync during week one</li>
              <li>• Hardware dispatch once payment clears</li>
            </ul>
            <textarea
              className="min-h-[140px] w-full rounded-2xl border border-white/15 bg-black/20 px-4 py-3 text-sm text-white placeholder:text-gray-500 focus:border-emerald-300 focus:outline-none"
              placeholder="Share field details or preferred onboarding dates (optional)"
            />
          </article>
        </div>
        <div className="flex flex-wrap gap-4">
          <button
            className="button-solid disabled:cursor-not-allowed disabled:opacity-60"
            disabled={isActivated}
            onClick={() => navigate('/payment', { state: { plan } })}
          >
            {isActivated ? 'Plan already activated' : 'Proceed to Payment'}
          </button>
          <button className="button-ghost" onClick={() => navigate('/')}>Choose another plan</button>
        </div>
        {isActivated && <p className="text-sm text-emerald-300">This plan is active. Pick a different tier if needed.</p>}
      </div>
    </div>
  )
}

function PaymentPage() {
  const navigate = useNavigate()
  const plan = usePlanSelection()
  const [paymentStatus, setPaymentStatus] = useState<'idle' | 'processing' | 'success'>('idle')
  const { markPlanPurchased, purchasedPlans } = usePurchase()
  const { user, login } = useUser()
  const alreadyActivated = purchasedPlans.includes(plan.id)

  useEffect(() => {
    if (paymentStatus !== 'success') return
    const timeout = window.setTimeout(() => {
      navigate('/', { replace: true })
    }, 1800)
    return () => window.clearTimeout(timeout)
  }, [navigate, paymentStatus])

  if (!user) {
    return (
      <div className="min-h-screen bg-gray-950 text-white">
        <div className="section-shell space-y-6 py-16 text-center">
          <h1 className="text-3xl font-semibold">Log in to complete payment</h1>
          <p className="section-copy">Sign in so we can capture the plan against your profile.</p>
          <div className="flex flex-wrap justify-center gap-4">
            <button className="button-solid" onClick={login}>
              Log in
            </button>
            <button className="button-ghost" onClick={() => navigate('/')}>Back to landing</button>
          </div>
        </div>
      </div>
    )
  }

  const handlePayment = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (alreadyActivated) {
      setPaymentStatus('success')
      return
    }
    setPaymentStatus('processing')
    setTimeout(() => {
      setPaymentStatus('success')
      markPlanPurchased(plan.id)
    }, 1200)
  }

  useEffect(() => {
    if (paymentStatus !== 'success') return
    const timeout = window.setTimeout(() => {
      navigate('/', { replace: true })
    }, 1800)
    return () => window.clearTimeout(timeout)
  }, [navigate, paymentStatus])

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <div className="section-shell space-y-8 py-16">
        <div className="flex items-center justify-between gap-4">
          <button className="text-sm text-gray-400 transition hover:text-white" onClick={() => navigate('/confirm', { state: { plan } })}>
            ← Back to confirmation
          </button>
          <p className="text-xs uppercase tracking-[0.4em] text-gray-500">Step 2 · Payment</p>
        </div>
        <div className="space-y-3 text-center">
          <h1 className="text-4xl font-semibold">Secure payment</h1>
          <p className="section-copy">Complete payment to activate the {plan.title} plan.</p>
        </div>
        <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
          <form className="glass-card space-y-5" onSubmit={handlePayment}>
            <div className="grid gap-4 md:grid-cols-2">
              <label className="space-y-2 text-sm text-gray-400">
                Full name
                <input
                  required
                  className="w-full rounded-2xl border border-white/15 bg-black/30 px-4 py-3 text-white placeholder:text-gray-500 focus:border-emerald-300 focus:outline-none"
                  placeholder="Aanya Sharma"
                  type="text"
                />
              </label>
              <label className="space-y-2 text-sm text-gray-400">
                Email
                <input
                  required
                  className="w-full rounded-2xl border border-white/15 bg-black/30 px-4 py-3 text-white placeholder:text-gray-500 focus:border-emerald-300 focus:outline-none"
                  placeholder="you@example.com"
                  type="email"
                />
              </label>
            </div>
            <label className="space-y-2 text-sm text-gray-400">
              Farm / organisation name
              <input
                className="w-full rounded-2xl border border-white/15 bg-black/30 px-4 py-3 text-white placeholder:text-gray-500 focus:border-emerald-300 focus:outline-none"
                placeholder="Sharma Agro Collective"
                type="text"
                required
              />
            </label>
            <div className="grid gap-4 md:grid-cols-2">
              <label className="space-y-2 text-sm text-gray-400">
                Card number
                <input
                  required
                  inputMode="numeric"
                  className="w-full rounded-2xl border border-white/15 bg-black/30 px-4 py-3 text-white placeholder:text-gray-500 focus:border-emerald-300 focus:outline-none"
                  placeholder="1234 5678 9012 3456"
                />
              </label>
              <div className="grid gap-4 md:grid-cols-2">
                <label className="space-y-2 text-sm text-gray-400">
                  Expiry
                  <input
                    required
                    className="w-full rounded-2xl border border-white/15 bg-black/30 px-4 py-3 text-white placeholder:text-gray-500 focus:border-emerald-300 focus:outline-none"
                    placeholder="MM / YY"
                  />
                </label>
                <label className="space-y-2 text-sm text-gray-400">
                  CVV
                  <input
                    required
                    className="w-full rounded-2xl border border-white/15 bg-black/30 px-4 py-3 text-white placeholder:text-gray-500 focus:border-emerald-300 focus:outline-none"
                    placeholder="123"
                    type="password"
                  />
                </label>
              </div>
            </div>
            <label className="space-y-2 text-sm text-gray-400">
              Notes for agronomist (optional)
              <textarea
                className="min-h-[120px] w-full rounded-2xl border border-white/15 bg-black/30 px-4 py-3 text-white placeholder:text-gray-500 focus:border-emerald-300 focus:outline-none"
                placeholder="Share acreage, crops, or preferred onboarding slots"
              />
            </label>
            <button
              className="button-solid w-full disabled:cursor-not-allowed disabled:opacity-60"
              disabled={paymentStatus === 'processing' || alreadyActivated}
              type="submit"
            >
              {alreadyActivated
                ? 'Plan already activated'
                : paymentStatus === 'processing'
                  ? 'Processing…'
                  : 'Pay & Activate Plan'}
            </button>
            {(paymentStatus === 'success' || alreadyActivated) && (
              <p className="text-center text-sm text-emerald-300">
                Payment successful! Your agronomist will reach out with deployment steps. Redirecting to the landing page…
              </p>
            )}
          </form>
          <article className="glass-card space-y-4">
            <p className="text-sm text-gray-400">Plan summary</p>
            <div>
              <p className="text-3xl font-semibold text-white">
                {plan.price}
                {plan.suffix && <span className="text-base font-medium text-gray-400">{plan.suffix}</span>}
              </p>
              <p className="text-sm text-gray-400">{plan.description}</p>
            </div>
            <ul className="space-y-2 text-sm text-gray-300">
              {plan.items.map((item) => (
                <li key={item} className="flex items-center gap-2">
                  <span className="text-emerald-400">•</span>
                  {item}
                </li>
              ))}
            </ul>
            <div className="rounded-2xl border border-white/10 bg-black/20 p-4 text-sm text-gray-400">
              Payments are encrypted end-to-end. Tax invoices and onboarding slots arrive immediately after success.
            </div>
            <button className="button-ghost w-full" onClick={() => navigate('/')}>Need to edit plan?</button>
          </article>
        </div>
      </div>
    </div>
  )
}

function AuthPage() {
  const navigate = useNavigate()
  const { setUserFromAuth } = useUser()
  const googleButtonRef = useRef<HTMLDivElement | null>(null)
  const [mode, setMode] = useState<'login' | 'signup'>('signup')
  const [formData, setFormData] = useState({ username: '', email: '', password: '' })
  const [users, setUsers] = useState<StoredUser[]>(() => getStoredUsers())
  const [status, setStatus] = useState<{ type: 'success' | 'error'; message: string } | null>(null)

  useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem(USERS_STORAGE_KEY, JSON.stringify(users))
  }, [users])

  const normalizedEmail = formData.email.trim().toLowerCase()

  const handleSignup = () => {
    if (!formData.username || !normalizedEmail || !formData.password) {
      setStatus({ type: 'error', message: 'Fill in username, email, and password.' })
      return
    }

    if (users.some((existing) => existing.email.toLowerCase() === normalizedEmail)) {
      setStatus({ type: 'error', message: 'An account with this email already exists.' })
      return
    }

    const newUser: StoredUser = {
      id:
        typeof crypto !== 'undefined' && 'randomUUID' in crypto
          ? crypto.randomUUID()
          : `user-${Date.now()}`,
      username: formData.username.trim(),
      email: normalizedEmail,
      password: formData.password,
      provider: 'password',
      createdAt: new Date().toISOString(),
    }

    const updated = [...users, newUser]
    setUsers(updated)
    void persistUsersToServer(updated)

    const profile: UserProfile = {
      id: newUser.id,
      name: newUser.username,
      avatar: avatarForName(newUser.username),
    }
    setUserFromAuth(profile)
    setStatus({ type: 'success', message: 'Account created! Profile saved to server.' })
    setTimeout(() => navigate('/'), 900)
  }

  const handleLogin = () => {
    if (!normalizedEmail || !formData.password) {
      setStatus({ type: 'error', message: 'Provide email and password to log in.' })
      return
    }
    const existing = users.find((item) => item.email.toLowerCase() === normalizedEmail)
    if (!existing) {
      setStatus({ type: 'error', message: 'No account found for this email. Sign up first.' })
      return
    }
    if (existing.provider === 'password' && existing.password !== formData.password) {
      setStatus({ type: 'error', message: 'Incorrect password. Try again.' })
      return
    }
    const profile: UserProfile = {
      id: existing.id,
      name: existing.username,
      avatar: avatarForName(existing.username),
    }
    setUserFromAuth(profile)
    setStatus({ type: 'success', message: 'Logged in successfully.' })
    setTimeout(() => navigate('/'), 600)
  }

  const handleFallbackGoogleLogin = () => {
    const baseName = formData.username.trim() || 'Google Farmer'
    const googleEmail = normalizedEmail || `farmer${Date.now()}@gmail.com`
    const existing = users.find((item) => item.email.toLowerCase() === googleEmail.toLowerCase())

    if (!existing) {
      const googleUser: StoredUser = {
        id: `google-${Date.now()}`,
        username: `${baseName} (Google)`,
        email: googleEmail.toLowerCase(),
        password: '',
        provider: 'google',
        createdAt: new Date().toISOString(),
      }
      const updated = [...users, googleUser]
      setUsers(updated)
      void persistUsersToServer(updated)
      const profile: UserProfile = {
        id: googleUser.id,
        name: googleUser.username,
        avatar: avatarForName(googleUser.username),
      }
      setUserFromAuth(profile)
    } else {
      const profile: UserProfile = {
        id: existing.id,
        name: existing.username,
        avatar: existing.avatarUrl || avatarForName(existing.username),
      }
      setUserFromAuth(profile)
    }
    setStatus({ type: 'success', message: 'Signed in with Google (fallback).' })
    setTimeout(() => navigate('/'), 700)
  }

  const handleGoogleCredential = useCallback(
    (response: { credential: string }) => {
      try {
        const [, payloadSegment] = response.credential.split('.')
        if (!payloadSegment) throw new Error('Bad Google credential payload')
        const normalizedPayload = payloadSegment.replace(/-/g, '+').replace(/_/g, '/')
        const payload = JSON.parse(atob(normalizedPayload)) as { name?: string; email?: string; picture?: string; sub?: string }
        const googleName = (payload.name ?? 'Google Farmer').trim()
        const emailValue = (payload.email ?? `${payload.sub ?? Date.now()}@gmail.com`).toLowerCase()
        let matchedUser: StoredUser | null = null
        let created = false
        setUsers((previous) => {
          const existing = previous.find((item) => item.email.toLowerCase() === emailValue)
          if (existing) {
            matchedUser = existing
            return previous
          }
          const googleUser: StoredUser = {
            id: `google-${payload.sub ?? Date.now()}`,
            username: googleName,
            email: emailValue,
            password: '',
            provider: 'google',
            createdAt: new Date().toISOString(),
            avatarUrl: payload.picture,
          }
          matchedUser = googleUser
          created = true
          const updated = [...previous, googleUser]
          void persistUsersToServer(updated)
          return updated
        })
        if (!matchedUser) {
          setStatus({ type: 'error', message: 'Unable to process Google credentials.' })
          return
        }
        const resolvedUser = matchedUser as StoredUser
        const profile: UserProfile = {
          id: resolvedUser.id,
          name: resolvedUser.username,
          avatar: resolvedUser.avatarUrl ?? avatarForName(resolvedUser.username),
        }
        setUserFromAuth(profile)
        setStatus({ type: 'success', message: created ? 'Google account connected.' : 'Signed in with Google.' })
        setTimeout(() => navigate('/'), 700)
      } catch (error) {
        console.warn('Google credential handling failed', error)
        setStatus({ type: 'error', message: 'Google sign-in failed. Use fallback button instead.' })
      }
    },
    [navigate, setUserFromAuth],
  )

  useEffect(() => {
    let cancelled = false
    loadGoogleIdentityScript()
      .then(() => {
        if (cancelled || !window.google) return
        window.google.accounts.id.initialize({ client_id: GOOGLE_CLIENT_ID, callback: handleGoogleCredential })
        if (googleButtonRef.current) {
          window.google.accounts.id.renderButton(googleButtonRef.current, {
            theme: 'outline',
            size: 'large',
            width: '100%',
          })
        }
        window.google.accounts.id.prompt()
      })
      .catch((error) => console.warn('Google Identity Services is unavailable', error))
    return () => {
      cancelled = true
    }
  }, [handleGoogleCredential])

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (mode === 'signup') {
      handleSignup()
    } else {
      handleLogin()
    }
  }

  const disabled = mode === 'signup'
    ? !formData.username || !formData.email || !formData.password
    : !formData.email || !formData.password

  return (
    <div className="min-h-screen bg-[#050608] px-4 py-10 text-gray-100">
      <div className="mx-auto flex min-h-[calc(100vh-80px)] w-full max-w-3xl flex-col justify-center">
        <div className="mb-6 flex justify-end">
          <button
            className="text-sm font-medium text-gray-400 transition hover:text-gray-100"
            onClick={() => navigate('/')}
            type="button"
          >
            ← Back to site
          </button>
        </div>
        <div className="rounded-lg border border-white/5 bg-[#0f1216] p-10 shadow-[0_25px_80px_-50px_rgba(0,0,0,0.85)]">
          <header className="mb-8 space-y-2">
            <p className="text-xs font-semibold uppercase tracking-[0.35em] text-emerald-200/60">Portal</p>
            <h1 className="text-3xl font-semibold text-white">Access Avec Agro</h1>
            <p className="text-sm text-gray-400">Secure sign-in for farmers, partners, and institutions.</p>
          </header>
          <div className="mb-8 flex flex-wrap items-center gap-4">
            <div className="inline-flex rounded-full border border-gray-700 bg-[#0c0f13] p-1">
              <button
                className={`rounded-full px-5 py-2 text-sm font-medium transition ${mode === 'signup' ? 'bg-[#2f8f5b] text-white shadow-sm' : 'text-gray-400 hover:text-gray-100'}`}
                onClick={() => setMode('signup')}
                type="button"
              >
                Sign up
              </button>
              <button
                className={`rounded-full px-5 py-2 text-sm font-medium transition ${mode === 'login' ? 'bg-[#2f8f5b] text-white shadow-sm' : 'text-gray-400 hover:text-gray-100'}`}
                onClick={() => setMode('login')}
                type="button"
              >
                Log in
              </button>
            </div>
          </div>
          <form className="space-y-5" onSubmit={handleSubmit}>
            <label className="block space-y-2 text-sm font-medium text-gray-200">
              Username
              <input
                className="w-full rounded-lg border border-gray-700 bg-transparent px-4 py-3 text-sm text-gray-100 placeholder:text-gray-600 focus:border-[#2f8f5b] focus:outline-none focus:ring-2 focus:ring-[#2f8f5b]/30"
                placeholder="Aanya Sharma"
                value={formData.username}
                onChange={(event) => setFormData((prev) => ({ ...prev, username: event.target.value }))}
              />
            </label>
            <label className="block space-y-2 text-sm font-medium text-gray-200">
              Email
              <input
                className="w-full rounded-lg border border-gray-700 bg-transparent px-4 py-3 text-sm text-gray-100 placeholder:text-gray-600 focus:border-[#2f8f5b] focus:outline-none focus:ring-2 focus:ring-[#2f8f5b]/30"
                placeholder="you@example.com"
                type="email"
                value={formData.email}
                onChange={(event) => setFormData((prev) => ({ ...prev, email: event.target.value }))}
              />
            </label>
            <label className="block space-y-2 text-sm font-medium text-gray-200">
              Password
              <input
                className="w-full rounded-lg border border-gray-700 bg-transparent px-4 py-3 text-sm text-gray-100 placeholder:text-gray-600 focus:border-[#2f8f5b] focus:outline-none focus:ring-2 focus:ring-[#2f8f5b]/30"
                placeholder="••••••••"
                type="password"
                value={formData.password}
                onChange={(event) => setFormData((prev) => ({ ...prev, password: event.target.value }))}
              />
            </label>
            <button
              className="w-full rounded-lg bg-[#2f8f5b] px-4 py-3 text-base font-semibold text-white shadow-sm transition hover:bg-[#2b7c4f] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2f8f5b] disabled:cursor-not-allowed disabled:opacity-60"
              disabled={disabled}
              type="submit"
            >
              {mode === 'signup' ? 'Create account' : 'Log in'}
            </button>
          </form>
          <div className="my-8 flex items-center gap-4 text-xs uppercase tracking-[0.4em] text-gray-600">
            <span className="h-px flex-1 bg-white/10" />
            or continue with
            <span className="h-px flex-1 bg-white/10" />
          </div>
          <div className="space-y-3">
            <div ref={googleButtonRef} className="flex justify-center" />
            <button
              className="w-full rounded-lg border border-gray-700 px-4 py-3 text-sm font-medium text-gray-100 transition hover:bg-gray-900/40"
              onClick={handleFallbackGoogleLogin}
              type="button"
            >
              Use fallback Google sign-in
            </button>
          </div>
          <p className="mt-6 text-xs text-gray-500">
            New registrations sync directly to Assets/login.json through the secure local Node service.
          </p>
          {status && (
            <p className={`mt-4 rounded-lg px-4 py-3 text-sm ${status.type === 'success' ? 'bg-emerald-500/10 text-emerald-300' : 'bg-red-500/10 text-red-300'}`}>
              {status.message}
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

function App() {
  const [user, setUser] = useState<UserProfile | null>(null)
  const [purchaseMap, setPurchaseMap] = useState<PurchaseMap>(() => getStoredPurchaseMap())

  useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem(PURCHASED_STORAGE_KEY, JSON.stringify(purchaseMap))
  }, [purchaseMap])

  useEffect(() => {
    if (typeof window === 'undefined') return
    const handleStorage = (event: StorageEvent) => {
      if (event.key === PURCHASED_STORAGE_KEY) {
        setPurchaseMap(getStoredPurchaseMap())
      }
    }
    window.addEventListener('storage', handleStorage)
    return () => window.removeEventListener('storage', handleStorage)
  }, [])

  const login = useCallback(() => {
    if (user) return user
    if (typeof window === 'undefined') return null
    const defaultName = 'Aanya Sharma'
    const name = window.prompt('Enter your farm or user name to continue', defaultName)?.trim()
    if (!name) return null
    const profile: UserProfile = {
      id: slugify(name) || `farmer-${Date.now()}`,
      name,
      avatar: avatarForName(name),
    }
    setUser(profile)
    return profile
  }, [user])

  const logout = useCallback(() => {
    setUser(null)
  }, [])

  const setUserFromAuth = useCallback((profile: UserProfile) => {
    setUser(profile)
  }, [])

  const markPlanPurchased = useCallback(
    (planId: string) => {
      if (!user) return
      setPurchaseMap((previous) => {
        const existing = previous[user.id] ?? []
        if (existing.includes(planId)) return previous
        return { ...previous, [user.id]: [...existing, planId] }
      })
    },
    [user],
  )

  const purchasedPlans = useMemo(() => {
    if (!user) return []
    return purchaseMap[user.id] ?? []
  }, [purchaseMap, user])

  const purchaseContextValue = useMemo<PurchaseContextValue>(
    () => ({
      purchasedPlans,
      markPlanPurchased,
    }),
    [markPlanPurchased, purchasedPlans],
  )

  const userContextValue = useMemo<UserContextValue>(
    () => ({
      user,
      login,
      logout,
      setUserFromAuth,
    }),
    [login, logout, setUserFromAuth, user],
  )

  return (
    <UserContext.Provider value={userContextValue}>
      <PurchaseContext.Provider value={purchaseContextValue}>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/confirm" element={<ConfirmationPage />} />
          <Route path="/payment" element={<PaymentPage />} />
          <Route path="/auth" element={<AuthPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </PurchaseContext.Provider>
    </UserContext.Provider>
  )
}

export default App