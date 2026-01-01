import { useEffect } from 'react';

export const usePushRegistration = (enabled) => {
  useEffect(() => {
    if (!enabled || !('serviceWorker' in navigator) || !('PushManager' in window)) return;

    Notification.requestPermission().then((permission) => {
      if (permission !== 'granted') return;
      navigator.serviceWorker.ready.then(async (registration) => {
        try {
          const subscription = await registration.pushManager.subscribe({
            userVisibleOnly: true,
            applicationServerKey: undefined
          });
          console.info('Push subscription', subscription);
        } catch (error) {
          console.warn('Push subscription failed', error);
        }
      });
    });
  }, [enabled]);
};
