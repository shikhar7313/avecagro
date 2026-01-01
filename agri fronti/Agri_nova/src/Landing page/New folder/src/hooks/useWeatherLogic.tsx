import { useState, useCallback } from 'react';
import { getLocationForWeather } from '../../../../utils/locationUtils'

type AnyObject = Record<string, unknown>;
type IntercroppingPrediction = Array<Record<string, unknown>> | Record<string, unknown> | null;

interface Answers {
    soilType?: string;
    lastCrop?: string;
    residueLeft?: string | boolean | number;
    soilMoisture?: string | number;
    region?: string;
    season?: string;
    soilPh?: string | number;
    [key: string]: unknown;
}

interface QuestionnaireEntry {
    savedAt: string;
    answers?: Answers;
}

interface WeatherData {
    temperature_C: number;
    humidity_percent: number;
    rainfall_mm: number;
    cloud_cover_percent: number;
    wind_speed_kmph: number;
    timezone: string;
}

const WEATHER_SERVER_URL = (import.meta as unknown as { env?: Record<string, string> }).env?.VITE_WEATHER_SERVER_URL ?? 'http://localhost:5005';
const QUESTIONNAIRE_SERVER_URL = 'http://localhost:5004';

function getErrorMessage(err: unknown): string {
    if (err instanceof Error) return err.message;
    try {
        return String(err);
    } catch {
        return 'Unknown error';
    }
}

export default function useWeatherLogic() {
    const [loading, setLoading] = useState<boolean>(true);
    const [npkPrediction, setNpkPrediction] = useState<AnyObject | null>(null);
    const [npkError, setNpkError] = useState<string | null>(null);
    const [intercroppingPrediction, setIntercroppingPrediction] = useState<IntercroppingPrediction>(null);
    const [intercroppingError, setIntercroppingError] = useState<string | null>(null);
    const [multiheightPrediction, setMultiheightPrediction] = useState<AnyObject | null>(null);
    const [multiheightError, setMultiheightError] = useState<string | null>(null);

    // ------------------------------------------
    // FETCH WEATHER + NPK FUNCTION
    // ------------------------------------------
    const fetchWeatherData = useCallback(async (): Promise<void> => {
        console.log('🌤️  [fetchWeatherData] Starting...');
        setLoading(true);
        try {
            // 1. Fetch questionnaire
            console.log('🌤️  [fetchWeatherData] Fetching questionnaire...');
            const qResponse = await fetch(`${QUESTIONNAIRE_SERVER_URL}/questionaire`);
            if (!qResponse.ok) throw new Error(`Questionnaire request failed: ${qResponse.status}`);
            const qText = await qResponse.text();
            const questionaireData = JSON.parse(qText);

            const latestEntry = Array.isArray(questionaireData)
                ? (questionaireData as QuestionnaireEntry[])
                      .sort((a, b) => new Date(b.savedAt).getTime() - new Date(a.savedAt).getTime())[0]
                : null;

            if (!latestEntry || !latestEntry.answers) {
                throw new Error('No questionnaire answers found');
            }

            // 2. Fetch weather from weather_server
            console.log('🌤️  [fetchWeatherData] Fetching weather from weather server...');
            const locationData = getLocationForWeather();
            const weatherResponse = await fetch(`${WEATHER_SERVER_URL}/api/get-weather`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    latitude: locationData.latitude,
                    longitude: locationData.longitude,
                    city: locationData.city,
                    name: locationData.name,
                }),
            });
            if (!weatherResponse.ok) throw new Error(`Weather fetch failed: ${weatherResponse.status}`);
            const weatherResult = await weatherResponse.json();
            if (weatherResult.status !== 'success' || !weatherResult.data) {
                throw new Error('Invalid weather response');
            }
            const weather = weatherResult.data as WeatherData;
            console.log('🌤️  [fetchWeatherData] ✅ Weather data:', weather);

            // 3. Prepare NPK request using weather_server
            console.log('🌤️  [fetchWeatherData] Preparing NPK request...');
            const npkPrepResponse = await fetch(`${WEATHER_SERVER_URL}/api/prepare-npk-request`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    soil_type: latestEntry.answers.soilType,
                    last_crop: latestEntry.answers.lastCrop,
                    residue_left: latestEntry.answers.residueLeft,
                    weather,
                }),
            });
            if (!npkPrepResponse.ok) throw new Error(`NPK request prep failed: ${npkPrepResponse.status}`);
            const npkPrepResult = await npkPrepResponse.json();
            if (npkPrepResult.status !== 'success' || !npkPrepResult.data) {
                throw new Error('Invalid NPK prep response');
            }
            const npkRequest = npkPrepResult.data;
            console.log('🌤️  [fetchWeatherData] NPK request payload:', npkRequest);

            // 4. Call NPK prediction endpoint
            console.log('🌤️  [fetchWeatherData] Calling NPK prediction...');
            const npkResponse = await fetch(`${QUESTIONNAIRE_SERVER_URL}/predictnpk`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(npkRequest),
            });

            if (!npkResponse.ok) throw new Error(`NPK prediction failed: ${npkResponse.status}`);
            const npkText = await npkResponse.text();
            const npkResult = JSON.parse(npkText);
            console.log('🌤️  [fetchWeatherData] ✅ NPK Prediction result:', npkResult);
            setNpkPrediction(npkResult);
        } catch (error: unknown) {
            const errMsg = getErrorMessage(error);
            console.error('❌ [fetchWeatherData] Error:', errMsg);
            setNpkError(errMsg);
        } finally {
            console.log('🌤️  [fetchWeatherData] Completed');
            setLoading(false);
        }
    }, []);

    // ------------------------------------------
    // FETCH NPK + INTERCROPPING
    // ------------------------------------------
    const fetchNpkAndIntercroppingData = useCallback(async (): Promise<void> => {
        console.log('🌾 [fetchNpkAndIntercroppingData] Starting...');
        setLoading(true);

        try {
            // 1. Fetch questionnaire
            console.log('🌾 [fetchNpkAndIntercroppingData] Fetching questionnaire...');
            const qResponse = await fetch(`${QUESTIONNAIRE_SERVER_URL}/questionaire`);
            if (!qResponse.ok) throw new Error('Questionnaire fetch failed');
            const qText = await qResponse.text();
            const questionaireData = JSON.parse(qText);
            const latestEntry = Array.isArray(questionaireData)
                ? (questionaireData as QuestionnaireEntry[])
                      .sort((a, b) => new Date(b.savedAt).getTime() - new Date(a.savedAt).getTime())[0]
                : null;

            if (!latestEntry || !latestEntry.answers) {
                throw new Error('No questionnaire answers found');
            }

            // 2. Fetch weather from weather_server
            console.log('🌾 [fetchNpkAndIntercroppingData] Fetching weather...');
            const locationData = getLocationForWeather();
            const weatherResponse = await fetch(`${WEATHER_SERVER_URL}/api/get-weather`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    latitude: locationData.latitude,
                    longitude: locationData.longitude,
                    city: locationData.city,
                    name: locationData.name,
                }),
            });
            if (!weatherResponse.ok) throw new Error('Weather fetch failed');
            const weatherResult = await weatherResponse.json();
            if (weatherResult.status !== 'success' || !weatherResult.data) {
                throw new Error('Invalid weather response');
            }
            const weather = weatherResult.data as WeatherData;
            console.log('🌾 [fetchNpkAndIntercroppingData] ✅ Weather data:', weather);

            // 3. Prepare and call NPK prediction
            console.log('🌾 [fetchNpkAndIntercroppingData] Preparing NPK request...');
            const npkPrepResponse = await fetch(`${WEATHER_SERVER_URL}/api/prepare-npk-request`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    soil_type: latestEntry.answers.soilType,
                    last_crop: latestEntry.answers.lastCrop,
                    residue_left: latestEntry.answers.residueLeft,
                    weather,
                }),
            });
            if (!npkPrepResponse.ok) throw new Error('NPK request prep failed');
            const npkPrepResult = await npkPrepResponse.json();
            if (npkPrepResult.status !== 'success') throw new Error('Invalid NPK prep response');

            const npkResponse = await fetch(`${QUESTIONNAIRE_SERVER_URL}/predictnpk`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(npkPrepResult.data),
            });
            if (!npkResponse.ok) throw new Error('NPK prediction failed');
            const npkText = await npkResponse.text();
            const npkResult = JSON.parse(npkText);
            console.log('🌾 [fetchNpkAndIntercroppingData] ✅ NPK result:', npkResult);
            setNpkPrediction(npkResult);

            // 4. Prepare and call intercropping prediction
            console.log('🌾 [fetchNpkAndIntercroppingData] Preparing intercropping request...');
            const interPrepResponse = await fetch(`${WEATHER_SERVER_URL}/api/prepare-intercropping-request`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    soil_type: latestEntry.answers.soilType,
                    last_crop: latestEntry.answers.lastCrop,
                    soil_n: npkResult?.estimated_N ?? 0,
                    soil_p: npkResult?.estimated_P ?? 0,
                    soil_k: npkResult?.estimated_K ?? 0,
                    soil_moisture: latestEntry.answers.soilMoisture,
                    weather,
                }),
            });
            if (!interPrepResponse.ok) throw new Error('Intercropping request prep failed');
            const interPrepResult = await interPrepResponse.json();
            if (interPrepResult.status !== 'success') throw new Error('Invalid intercropping prep response');

            console.log('🌾 [fetchNpkAndIntercroppingData] Calling intercropping prediction...');
            const interResponse = await fetch(`${QUESTIONNAIRE_SERVER_URL}/predict_intercropping`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(interPrepResult.data),
            });
            if (!interResponse.ok) throw new Error('Intercropping prediction failed');
            const interText = await interResponse.text();
            const interResult = JSON.parse(interText);
            console.log('🌾 [fetchNpkAndIntercroppingData] ✅ Intercropping result (raw):', interResult);

            const normalizedIntercropping = Array.isArray(interResult?.raw_results)
                ? interResult.raw_results
                : Array.isArray(interResult?.simplified_results)
                    ? interResult.simplified_results
                    : Array.isArray(interResult)
                        ? interResult
                        : [];

            console.log('🌾 [fetchNpkAndIntercroppingData] ✅ Intercropping result (normalized array):', normalizedIntercropping);
            setIntercroppingPrediction(normalizedIntercropping);

        } catch (err: unknown) {
            const errMsg = getErrorMessage(err);
            console.error('❌ [fetchNpkAndIntercroppingData] Error:', errMsg);
            setIntercroppingError(errMsg);
        } finally {
            console.log('🌾 [fetchNpkAndIntercroppingData] Completed');
            setLoading(false);
        }
    }, []);

    // ------------------------------------------
    // MULTIHEIGHT FUNCTION
    // ------------------------------------------
    const fetchMultiheightData = useCallback(async (): Promise<void> => {
        console.log('📋 [fetchMultiheightData] Starting...');
        setLoading(true);

        try {
            // 1. Fetch questionnaire
            console.log('📋 [fetchMultiheightData] Fetching questionnaire...');
            const qResponse = await fetch(`${QUESTIONNAIRE_SERVER_URL}/questionaire`);
            if (!qResponse.ok) throw new Error('Questionnaire fetch failed');
            const qText = await qResponse.text();
            const data = JSON.parse(qText);
            const latestEntry = Array.isArray(data)
                ? (data as QuestionnaireEntry[]).sort((a, b) => new Date(b.savedAt).getTime() - new Date(a.savedAt).getTime())[0]
                : null;

            if (!latestEntry || !latestEntry.answers) {
                throw new Error('No questionnaire answers found');
            }

            // 2. Fetch weather from weather_server
            console.log('📋 [fetchMultiheightData] Fetching weather...');
            const locationData = getLocationForWeather();
            const weatherResponse = await fetch(`${WEATHER_SERVER_URL}/api/get-weather`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    latitude: locationData.latitude,
                    longitude: locationData.longitude,
                    city: locationData.city,
                    name: locationData.name,
                }),
            });
            if (!weatherResponse.ok) throw new Error('Weather fetch failed');
            const weatherResult = await weatherResponse.json();
            if (weatherResult.status !== 'success' || !weatherResult.data) {
                throw new Error('Invalid weather response');
            }
            const weather = weatherResult.data as WeatherData;
            console.log('📋 [fetchMultiheightData] ✅ Weather data:', weather);

            // 3. Prepare and call multiheight prediction
            console.log('📋 [fetchMultiheightData] Preparing multiheight request...');
            const multiPrepResponse = await fetch(`${WEATHER_SERVER_URL}/api/prepare-multiheight-request`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    soil_type: latestEntry.answers.soilType,
                    last_crop: latestEntry.answers.lastCrop,
                    region: latestEntry.answers.region,
                    season: latestEntry.answers.season,
                    soil_ph: latestEntry.answers.soilPh,
                    weather,
                }),
            });
            if (!multiPrepResponse.ok) throw new Error('Multiheight request prep failed');
            const multiPrepResult = await multiPrepResponse.json();
            if (multiPrepResult.status !== 'success') throw new Error('Invalid multiheight prep response');

            console.log('📋 [fetchMultiheightData] Calling multiheight prediction...');
            const res = await fetch(`${QUESTIONNAIRE_SERVER_URL}/predict_multiheight`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(multiPrepResult.data),
            });
            if (!res.ok) throw new Error('Multiheight prediction failed');
            const resText = await res.text();
            const result = JSON.parse(resText);
            console.log('📋 [fetchMultiheightData] ✅ Multiheight result:', result);
            setMultiheightPrediction(result);

        } catch (err: unknown) {
            const errMsg = getErrorMessage(err);
            console.error('❌ [fetchMultiheightData] Error:', errMsg);
            setMultiheightError(errMsg);
        } finally {
            console.log('📋 [fetchMultiheightData] Completed');
            setLoading(false);
        }
    }, []);

    // Note: Predictions are triggered manually from the event listener in App.tsx
    // The hook does NOT run predictions on mount - it only exports the fetch functions

    return {
        loading,
        npkPrediction,
        npkError,
        intercroppingPrediction,
        intercroppingError,
        multiheightPrediction,
        multiheightError,
        fetchWeatherData,
        fetchNpkAndIntercroppingData,
        fetchMultiheightData,
    } as const;
}
