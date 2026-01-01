import { useEffect, useRef } from 'react';

export const useAutoScroll = (dependency) => {
  const containerRef = useRef(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTo({
        top: containerRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  }, [dependency]);

  return containerRef;
};
