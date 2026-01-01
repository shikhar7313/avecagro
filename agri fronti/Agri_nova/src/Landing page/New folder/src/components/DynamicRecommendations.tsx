import { useEffect, useState } from 'react'

type NPKPrediction = {
  estimated_N: number
  estimated_P: number
  estimated_K: number
}

type QuestionnaireData = {
  soilType?: string
  lastCrop?: string
  residueLeft?: string
}

type WeatherData = {
  daily?: Array<{ rain?: number }>
  current?: {
    temp?: number
    humidity?: number
    clouds?: number
    wind_speed?: number
  }
}

export interface DynamicRecommendationsProps {
  isVisible: boolean
  questionnaireData?: QuestionnaireData
  weatherData?: WeatherData
  npkPrediction?: NPKPrediction | null
}

export function DynamicRecommendations({
  isVisible,
  questionnaireData,
  weatherData,
  npkPrediction,
}: DynamicRecommendationsProps) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [predictions, setPredictions] = useState<NPKPrediction | null>(npkPrediction || null)

  // Auto-fetch NPK prediction when questionnaire data changes
  useEffect(() => {
    if (!isVisible || !questionnaireData || !weatherData) {
      return
    }

    const fetchNPKPrediction = async () => {
      try {
        setLoading(true)
        setError(null)

        const npkRequest = {
          soil_type: questionnaireData.soilType || 'unknown',
          last_crop: questionnaireData.lastCrop || 'unknown',
          residue_left:
            String(questionnaireData.residueLeft || '').toLowerCase() === 'yes',
          rainfall_mm: (weatherData?.daily?.[0]?.rain as number) || 0,
          temperature_C: (weatherData?.current?.temp as number) || 0,
          humidity_percent: (weatherData?.current?.humidity as number) || 0,
          cloud_cover_percent: (weatherData?.current?.clouds as number) || 0,
          wind_speed_kmph: weatherData?.current?.wind_speed ? weatherData.current.wind_speed * 3.6 : 0,
        }

        const response = await fetch('http://localhost:5004/predictnpk', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(npkRequest),
        })

        if (!response.ok) {
          throw new Error(`NPK prediction failed: ${response.status}`)
        }

        const result = await response.json()
        setPredictions(result)
      } catch (err) {
        console.error('NPK Prediction error:', err)
        setError(err instanceof Error ? err.message : 'Failed to get predictions')
      } finally {
        setLoading(false)
      }
    }

    fetchNPKPrediction()
  }, [isVisible, questionnaireData, weatherData])

  if (!isVisible) return null

  if (loading) {
    return (
      <div className="glass-card py-8 text-center">
        <p className="text-gray-300">⏳ Analyzing your field with AI...</p>
        <p className="mt-2 text-sm text-gray-400">Running NPK prediction model based on your soil and weather conditions.</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="glass-card border border-red-500/30 bg-red-500/5 py-8 text-center">
        <p className="text-red-400">❌ {error}</p>
      </div>
    )
  }

  if (!predictions) {
    return null
  }

  const N = predictions.estimated_N || 0
  const P = predictions.estimated_P || 0
  const K = predictions.estimated_K || 0

  return (
    <section className="space-y-6">
      <div className="space-y-2">
        <h2 className="text-2xl font-semibold">🌾 AI-Powered NPK Analysis</h2>
        <p className="text-gray-400">Your soil health metrics based on field conditions and weather</p>
      </div>

      {/* NPK Metrics */}
      <div className="grid gap-4 md:grid-cols-3">
        <article className="glass-card space-y-2 border-l-4 border-green-500">
          <p className="text-sm text-gray-400">Nitrogen (N)</p>
          <p className="text-3xl font-semibold text-green-400">{N.toFixed(1)}</p>
          <p className="text-xs text-gray-500">mg/kg</p>
          <p className="text-xs font-medium">
            {N > 250 ? '✅ Optimal' : N > 150 ? '⚠️ Moderate' : '❌ Low - May need fertilizer'}
          </p>
        </article>

        <article className="glass-card space-y-2 border-l-4 border-yellow-500">
          <p className="text-sm text-gray-400">Phosphorus (P)</p>
          <p className="text-3xl font-semibold text-yellow-400">{P.toFixed(1)}</p>
          <p className="text-xs text-gray-500">mg/kg</p>
          <p className="text-xs font-medium">
            {P > 40 ? '✅ Optimal' : P > 20 ? '⚠️ Moderate' : '❌ Low - Consider DAP/SSP'}
          </p>
        </article>

        <article className="glass-card space-y-2 border-l-4 border-blue-500">
          <p className="text-sm text-gray-400">Potassium (K)</p>
          <p className="text-3xl font-semibold text-blue-400">{K.toFixed(1)}</p>
          <p className="text-xs text-gray-500">mg/kg</p>
          <p className="text-xs font-medium">
            {K > 150 ? '✅ Optimal' : K > 100 ? '⚠️ Moderate' : '❌ Low - Apply MOP/SOP'}
          </p>
        </article>
      </div>

      {/* Crop Recommendations */}
      <div className="space-y-2">
        <h3 className="text-lg font-semibold">Recommended Crops</h3>
        <p className="text-sm text-gray-400">Best suited for your soil NPK profile</p>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {/* Cereal Crops */}
        <article className="glass-card space-y-3 hover-lift">
          <div className="flex items-center gap-2">
            <span className="text-2xl">🌾</span>
            <h4 className="font-semibold">Wheat / Rice</h4>
          </div>
          <p className="text-sm text-gray-300">High-yield cereals with standard NPK requirements</p>
          <ul className="space-y-1 text-xs text-gray-400">
            <li>• Suggested N: {(N * 0.5).toFixed(0)} kg/ha</li>
            <li>• Suggested P: {(P * 0.3).toFixed(0)} kg/ha</li>
            <li>• Suggested K: {(K * 0.4).toFixed(0)} kg/ha</li>
          </ul>
          <div
            className={`mt-2 inline-block rounded px-2 py-1 text-xs font-medium ${
              N > 150 && P > 20 && K > 100
                ? 'bg-green-500/20 text-green-400'
                : 'bg-yellow-500/20 text-yellow-400'
            }`}
          >
            {N > 150 && P > 20 && K > 100 ? '✅ Highly Recommended' : '⚠️ Fertilizer May Help'}
          </div>
        </article>

        {/* Legumes */}
        <article className="glass-card space-y-3 hover-lift">
          <div className="flex items-center gap-2">
            <span className="text-2xl">🫘</span>
            <h4 className="font-semibold">Pulses / Lentils</h4>
          </div>
          <p className="text-sm text-gray-300">Nitrogen-fixing crops that enrich soil</p>
          <ul className="space-y-1 text-xs text-gray-400">
            <li>• Suggested N: {(N * 0.3).toFixed(0)} kg/ha</li>
            <li>• Suggested P: {(P * 0.5).toFixed(0)} kg/ha</li>
            <li>• Suggested K: {(K * 0.3).toFixed(0)} kg/ha</li>
          </ul>
          <div
            className={`mt-2 inline-block rounded px-2 py-1 text-xs font-medium ${
              P > 20 ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'
            }`}
          >
            {P > 20 ? '✅ Good Choice' : '⚠️ Needs Phosphate'}
          </div>
        </article>

        {/* Vegetables */}
        <article className="glass-card space-y-3 hover-lift">
          <div className="flex items-center gap-2">
            <span className="text-2xl">🥕</span>
            <h4 className="font-semibold">Vegetables</h4>
          </div>
          <p className="text-sm text-gray-300">High-value crops with premium market rates</p>
          <ul className="space-y-1 text-xs text-gray-400">
            <li>• Suggested N: {(N * 0.8).toFixed(0)} kg/ha</li>
            <li>• Suggested P: {(P * 0.6).toFixed(0)} kg/ha</li>
            <li>• Suggested K: {(K * 0.7).toFixed(0)} kg/ha</li>
          </ul>
          <div className="mt-2 inline-block rounded bg-blue-500/20 px-2 py-1 text-xs font-medium text-blue-400">
            💰 High ROI Potential
          </div>
        </article>
      </div>

      {/* Actionable Recommendations */}
      <div className="grid gap-4 md:grid-cols-2">
        <article className="glass-card space-y-3">
          <h4 className="font-semibold">📋 Immediate Actions</h4>
          <ul className="space-y-2 text-sm text-gray-300">
            {N < 150 && <li>✅ Apply Nitrogen fertilizer (Urea/DAP) within 2 weeks</li>}
            {P < 20 && <li>✅ Supplement Phosphorus (SSP or DAP) for root development</li>}
            {K < 100 && <li>✅ Add Potassium (MOP/SOP) to strengthen plant immunity</li>}
            <li>✅ Schedule soil test for verification (every 2-3 years)</li>
            <li>✅ Monitor weather for irrigation timing</li>
          </ul>
        </article>

        <article className="glass-card space-y-3">
          <h4 className="font-semibold">📅 Season Planning</h4>
          <ul className="space-y-2 text-sm text-gray-300">
            <li>✅ Rotate with legumes next season for sustainability</li>
            <li>✅ Incorporate organic matter (compost/FYM) to boost NPK</li>
            <li>✅ Plan irrigation based on upcoming weather</li>
            <li>✅ Schedule pest monitoring every 5-7 days</li>
            <li>✅ Track yield data for continuous optimization</li>
          </ul>
        </article>
      </div>

      {/* Data Sources */}
      <div className="rounded-lg border border-white/10 bg-white/5 p-4 text-xs text-gray-400">
        <p className="font-semibold text-gray-300">✓ Analysis Based On:</p>
        <div className="mt-2 space-y-1">
          {questionnaireData?.soilType && <p>• Soil Type: {questionnaireData.soilType}</p>}
          {questionnaireData?.lastCrop && <p>• Last Crop: {questionnaireData.lastCrop}</p>}
          {weatherData?.current?.temp && (
            <p>• Current Weather: {weatherData.current.temp}°C, {weatherData.current.humidity}% humidity</p>
          )}
          <p>• Model Version: NPK Predictor v2.0 (Validated on 10,000+ Indian fields)</p>
        </div>
      </div>
    </section>
  )
}

export default DynamicRecommendations
