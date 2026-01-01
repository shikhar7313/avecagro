import { useEffect, useRef, useState } from 'react';

/**
 * useSpatialAudio hook
 * @param {string} audioUrl - URL or path to the audio file
 * @param {React.RefObject} cameraRef - ref object to Three.js camera
 * @returns {AnalyserNode|null} - AnalyserNode for frequency data
 */
export default function useSpatialAudio(audioUrl, cameraRef) {
  const analyserRef = useRef(null);
  const bufferRef = useRef(null);
  const contextRef = useRef(null);
  const pannerRef = useRef(null);
  const audioElRef = useRef(null);
  const [analyserNode, setAnalyserNode] = useState(null);

  useEffect(() => {
    if (!audioUrl) return;
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const context = new AudioContext();
    contextRef.current = context;
    const panner = context.createPanner();
    panner.panningModel = 'HRTF';
    panner.distanceModel = 'inverse';
    panner.connect(context.destination);

    const analyser = context.createAnalyser();
    panner.connect(analyser);
    analyser.connect(context.destination);
    analyserRef.current = analyser;
    setAnalyserNode(analyser); // update state so components rerender with analyser
    pannerRef.current = panner;

    // update listener position
    const updateListener = () => {
      const cam = cameraRef.current;
      if (cam) {
        const pos = cam.position;
        context.listener.setPosition(pos.x, pos.y, pos.z);
      }
      requestAnimationFrame(updateListener);
    };
    updateListener();

    // set up media element source for reliable playback
    const audioEl = new Audio(audioUrl);
    audioEl.crossOrigin = 'anonymous';
    audioEl.loop = true; // enable looping for testing
    audioEl.volume = 1.0;
    audioEl.preload = 'auto';
    audioEl.style.display = 'none';
    document.body.appendChild(audioEl);
    audioEl.load();
    audioElRef.current = audioEl;
    const sourceNode = context.createMediaElementSource(audioEl);
    sourceNode.connect(panner);
    // also load buffer for fallback playback
    fetch(audioUrl)
      .then(res => res.arrayBuffer())
      .then(data => context.decodeAudioData(data))
      .then(buffer => {
        bufferRef.current = buffer;
        console.log('Audio buffer decoded for fallback');
      })
      .catch(e => console.warn('Buffer decode failed', e));

    return () => {
      context.close();
    };
  }, [audioUrl, cameraRef]);

  // play the loaded buffer once
  const play = () => {
    const context = contextRef.current;
    const audioEl = audioElRef.current;
    const buffer = bufferRef.current;
    // Ensure context is resumed before playing
    const playBufferFallback = () => {
      if (buffer) {
        const src = context.createBufferSource();
        src.buffer = buffer;
        src.connect(pannerRef.current);
        src.start(0);
      }
    };
    if (context) {
      if (context.state === 'suspended') {
        context.resume()
          .then(() => {
            if (audioEl) {
              audioEl.play().catch(e => {
                console.warn('AudioElement play failed, fallback to buffer', e);
                playBufferFallback();
              });
            } else {
              playBufferFallback();
            }
          })
          .catch(e => {
            console.warn('context resume failed', e);
            playBufferFallback();
          });
        return;
      } else {
        if (audioEl) {
          audioEl.play().catch(e => {
            console.warn('AudioElement play failed, fallback to buffer', e);
            playBufferFallback();
          });
          return;
        }
        playBufferFallback();
      }
    }
  };

  return [analyserNode, play];
}
