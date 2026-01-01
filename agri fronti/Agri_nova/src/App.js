import React, { useState, useEffect, useRef, Suspense } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Dashboard from './components/Dashboard';
import Chatbot from './components/Chatbot';
import LoanManagement from './components/LoanManagement';
import Weather from './components/Weather';
import SoilWater from './components/SoilWater';
import PestManagement from './components/PestManagement';
import ReportsAnalytics from './components/ReportsAnalytics';
import EquipmentManagement from './components/EquipmentManagement';
import IotDevices from './components/IotDevices';
import Settings from './components/Settings';
import CommunityChatEmbed from './components/CommunityChatEmbed';
import tasksData from './data/dashboard/tasks.json';
import weatherData from './data/dashboard/weather.json';
import { generateAlerts, generateRecentActivities } from './utils/dataGenerators';
import { getLocationForWeather } from './utils/locationUtils';
import BufferPage from './components/BufferPage';
import { storeQuestionnaireFarmSize, getCurrentUserFarmSize } from './utils/userDataUtils';

const QUESTIONNAIRE_API_BASE = process.env.REACT_APP_QUESTIONNAIRE_API || 'http://localhost:7001';

function parseAcreageLabel(label) {
  if (!label) return 0;
  const normalized = label.toString().toLowerCase();
  const matches = normalized.match(/\d+(\.\d+)?/g);
  if (matches && matches.length >= 2) {
    const first = parseFloat(matches[0]);
    const last = parseFloat(matches[matches.length - 1]);
    if (!isNaN(first) && !isNaN(last)) {
      return (first + last) / 2;
    }
  }
  if (matches && matches.length === 1) {
    const value = parseFloat(matches[0]);
    if (!isNaN(value)) {
      if (normalized.includes('<')) {
        return value / 2;
      }
      return value;
    }
  }
  if (normalized.includes('+') && matches && matches.length > 0) {
    const base = parseFloat(matches[0]);
    return isNaN(base) ? 0 : base;
  }
  const lookup = {
    small: 1.5,
    medium: 4,
    large: 8,
  };
  const found = Object.entries(lookup).find(([key]) => normalized.includes(key));
  return found ? found[1] : 0;
}

// Your video URLs, **make sure paths are correct and accessible**
const videos = [
  '/buffer_video/vid1.mp4',
  '/buffer_video/vid4.mp4',
];

function App() {
  const [loading, setLoading] = useState(true);
  const [loadDuration, setLoadDuration] = useState(5000); // default 5s
  const [selectedVideo, setSelectedVideo] = useState(null);

  useEffect(() => {
    // Pick a random video on mount
    const randomIndex = Math.floor(Math.random() * videos.length);
    setSelectedVideo(videos[randomIndex]);

    // Start timer immediately
    const startTime = performance.now();

    // We’ll consider the app loaded after all assets load + a min duration (optional)
    const handleLoadComplete = () => {
      const endTime = performance.now();
      const duration = endTime - startTime;
      // Minimum duration so loading bar always visible (e.g., 2s)
      const finalDuration = Math.max(duration, 2000);
      setLoadDuration(finalDuration);

      // Wait for loading bar animation to finish then hide loader
      setTimeout(() => {
        setLoading(false);
      }, finalDuration);
    };

    // Listen for window load event to detect when page is fully loaded
    if (document.readyState === 'complete') {
      // Already loaded
      handleLoadComplete();
    } else {
      window.addEventListener('load', handleLoadComplete);
      return () => window.removeEventListener('load', handleLoadComplete);
    }
  }, []);

  // API base: use env variable or default to proxy
  const API_BASE = process.env.REACT_APP_API_BASE || '';
  const mainRef = useRef(null);
  // Dashboard data slices
  const [alerts, setAlerts] = useState([]);
  const [recentActivities, setRecentActivities] = useState([]);
  const [tasks, setTasks] = useState([]);
  const [weather, setWeather] = useState({});
  const [activeTab, setActiveTab] = useState('dashboard');
  const [farmSizeAcres, setFarmSizeAcres] = useState(() => getCurrentUserFarmSize());

  // Handler to reset persisted data and reload defaults
  const handleReset = () => {
    localStorage.removeItem('tasks');
    // reset tasks and regenerate alerts & recent activities
    setTasks(tasksData);
    const regeneratedAlerts = generateAlerts(tasksData, weatherData);
    setAlerts(regeneratedAlerts);
    const regeneratedActivities = generateRecentActivities(tasksData);
    setRecentActivities(regeneratedActivities);
  };


  useEffect(() => {
    // Fetch tasks and dynamic weather from server API
    fetch(`${API_BASE}/api/tasks`)
      .then((res) => res.json())
      .then((tasksList) => {
        setTasks(tasksList);

        // Fetch live weather based on stored location or fallback to Delhi
        const loadWeather = async () => {
          try {
            const locationData = getLocationForWeather();
            console.log('App: Loading weather with location:', locationData);

            const res = await fetch(`${API_BASE}/api/weather${locationData.query}`);
            const weatherAPI = await res.json();
            const dynamicWeather = { major: weatherAPI.major };
            setWeather(dynamicWeather);
            setAlerts(generateAlerts(tasksList, dynamicWeather));
          } catch (err) {
            console.error('Failed to load weather:', err);
          } finally {
            setRecentActivities(generateRecentActivities(tasksList));
          }
        };

        loadWeather();
      })
      .catch((err) => console.error('Failed to load tasks:', err));
  }, []);

  useEffect(() => {
    const username = localStorage.getItem('username');
    if (!username) return;

    const controller = new AbortController();

    const loadFarmProfile = async () => {
      try {
        const url = new URL('/api/questionnaire/latest', QUESTIONNAIRE_API_BASE);
        url.searchParams.set('username', username);
        const response = await fetch(url.toString(), { signal: controller.signal });
        if (!response.ok) {
          throw new Error('Failed to load questionnaire profile');
        }
        const data = await response.json();
        const acreageLabel = data?.entry?.answers?.acreage;
        if (!acreageLabel) return;
        const acres = parseAcreageLabel(acreageLabel);
        if (acres <= 0) return;
        storeQuestionnaireFarmSize(acres);
        setFarmSizeAcres(acres);
      } catch (error) {
        if (error.name !== 'AbortError') {
          console.error('Failed to derive farm size from questionnaire', error);
        }
      }
    };

    loadFarmProfile();
    return () => controller.abort();
  }, []);

  // Listen for location updates to refresh weather
  useEffect(() => {
    const handleLocationUpdate = async () => {
      try {
        const locationData = getLocationForWeather();
        console.log('App: Refreshing weather after location update:', locationData);

        const res = await fetch(`${API_BASE}/api/weather${locationData.query}`);
        const weatherAPI = await res.json();
        const dynamicWeather = { major: weatherAPI.major };
        setWeather(dynamicWeather);

        // Regenerate alerts with new weather data
        setAlerts(generateAlerts(tasks, dynamicWeather));
      } catch (err) {
        console.error('Failed to refresh weather after location update:', err);
      }
    };

    window.addEventListener('userLocationUpdated', handleLocationUpdate);
    return () => window.removeEventListener('userLocationUpdated', handleLocationUpdate);
  }, [tasks]);

  // No localStorage persistence: server handles data

  useEffect(() => {
    gsap.registerPlugin(ScrollTrigger);
    // animate elements marked with data-scroll
    if (mainRef.current) {
      gsap.utils.toArray('[data-scroll]').forEach((el) => {
        gsap.from(el, {
          opacity: 0,
          y: 50,
          duration: 1,
          scrollTrigger: {
            trigger: el,
            scroller: mainRef.current,
            start: 'top 80%',
            toggleActions: 'play none none reset',
          },
        });
      });

      // horizontal scroll sections
      gsap.utils.toArray('[data-horizontal]').forEach((section) => {
        const slider = section.querySelector('.horizontal-slider');
        if (!slider) return;
        ScrollTrigger.create({
          trigger: section,
          scroller: mainRef.current,
          start: 'top top',
          end: () => `+=${slider.scrollWidth - window.innerWidth}`,
          scrub: true,
          pin: true,
          anticipatePin: 1,
          onUpdate: (self) => {
            gsap.to(slider, { x: -((slider.scrollWidth - window.innerWidth) * self.progress), ease: 'none' });
          },
        });
      });

      // 3D Parallax layers
      gsap.utils.toArray('[data-depth]').forEach((layer) => {
        const depth = parseFloat(layer.getAttribute('data-depth')) || 0;
        ScrollTrigger.create({
          scroller: mainRef.current,
          trigger: mainRef.current,
          start: 'top top',
          end: 'bottom bottom',
          scrub: true,
          onUpdate: (self) => {
            gsap.to(layer, {
              y: -(self.progress * depth * window.innerHeight),
              ease: 'none',
            });
          },
        });
      });
      // (ScrollTrigger cleanup handled below)

      // clip-path / shape morphing on scroll
      gsap.utils.toArray('[data-morph]').forEach((el) => {
        const startClip = el.getAttribute('data-clip-start') || 'polygon(0 0, 100% 0, 100% 100%, 0% 100%)';
        const endClip = el.getAttribute('data-clip-end') || 'polygon(50% 0, 100% 50%, 50% 100%, 0% 50%)';
        gsap.fromTo(el,
          { clipPath: startClip },
          {
            clipPath: endClip,
            ease: 'none',
            scrollTrigger: {
              trigger: el,
              scroller: mainRef.current,
              start: 'top 90%',
              end: 'bottom 10%',
              scrub: true,
            }
          }
        );
      });


      // element pinning
      gsap.utils.toArray('[data-pin]').forEach((el) => {
        const pinEndValue = el.getAttribute('data-pin-end');
        ScrollTrigger.create({
          trigger: el,
          scroller: mainRef.current,
          start: 'top top',
          end: pinEndValue ? `+=${pinEndValue}` : 'bottom top',
          pin: true,
          pinSpacing: true,
        });
      });
    }
  }, []);

  // perspective shifts on mousemove - DISABLED for all pages
  useEffect(() => {
    const container = mainRef.current;
    if (!container) return;

    // Reset any existing transforms to ensure no tilt effect
    gsap.to(container, {
      rotationY: 0,
      rotationX: 0,
      transformPerspective: 'none',
      ease: 'power1.out',
      duration: 0.3,
    });

    // Tilt effect completely disabled for all pages
    // No mouse move listeners added

  }, [activeTab]);

  // Only show BufferPage while loading and video is selected
  if (loading && selectedVideo) {
    return <BufferPage videoSrc={selectedVideo} duration={loadDuration} />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-emerald-900 relative overflow-hidden">
      {/* Three.js Background with suspense */}
      <Suspense fallback={null}>
      </Suspense>

      {/* Main Layout */}
      <div className="flex h-screen relative z-10">
        {/* Sidebar */}
        <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} farmSizeAcres={farmSizeAcres} />

        {/* Main Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Header */}
          <Header
            data={{ weather }}
            farmSizeAcres={farmSizeAcres}
            showIoT={typeof farmSizeAcres === 'number' ? farmSizeAcres >= 2 : true}
            showDrone={typeof farmSizeAcres === 'number' ? farmSizeAcres >= 7 : true}
          />

          {/* Dashboard Content */}
          <main ref={mainRef} className="flex-1 overflow-x-hidden overflow-y-auto min-h-0 flex flex-col">
            {activeTab === 'dashboard' && (
              <Dashboard
                alerts={alerts}
                setAlerts={setAlerts}
                recentActivities={recentActivities}
                setRecentActivities={setRecentActivities}
                tasks={tasks}
                setTasks={setTasks}
                weather={weather}
                setWeather={setWeather}
              />
            )}
            {activeTab === 'chatbot' && (
              <Chatbot />
            )}
            {activeTab === 'loan-management' && (
              <LoanManagement />
            )}
            {activeTab === 'weather' && (
              <Weather />
            )}
            {activeTab === 'soil-water' && (
              <SoilWater />
            )}
            {activeTab === 'pest-management' && (
              <PestManagement />
            )}
            {activeTab === 'reports' && (
              <ReportsAnalytics />
            )}
            {activeTab === 'equipment' && (
              <EquipmentManagement />
            )}
            {activeTab === 'iot-devices' && (
              <IotDevices />
            )}
            {activeTab === 'community' && (
              <CommunityChatEmbed />
            )}
            {activeTab === 'settings' && (
              <Settings />
            )}
            {/* No fallback placeholder */}
          </main>
        </div>
      </div>
      {/* Fullscreen overlay to hide native cursor 
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        cursor: 'none',
        pointerEvents: 'none',
        zIndex: 10001
      }} />*/}
      {/* Futuristic custom cursor component */}
      {/* <CustomCursor /> */}
    </div>
  );
}

export default App;
