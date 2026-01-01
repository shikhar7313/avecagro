import { useEffect, useRef } from 'react';

const PLACEHOLDER_ID = 'google_translate_element';

const GoogleTranslate = () => {
  const containerRef = useRef(null);

  useEffect(() => {
    const target = containerRef.current;
    if (!target) return;

    const placeholder = document.getElementById(PLACEHOLDER_ID);
    if (!placeholder) return;

    const moveWidgetIntoTarget = () => {
      const hasWidget = placeholder.querySelector('.goog-te-combo');
      if (!hasWidget) return false;

      placeholder.style.display = 'block';
      if (placeholder.parentElement !== target) {
        target.innerHTML = '';
        target.appendChild(placeholder);
      }
      return true;
    };

    if (moveWidgetIntoTarget()) {
      return () => {
        placeholder.style.display = 'none';
        if (placeholder.parentElement !== document.body) {
          document.body.appendChild(placeholder);
        }
      };
    }

    const interval = setInterval(() => {
      if (moveWidgetIntoTarget()) {
        clearInterval(interval);
      }
    }, 400);

    return () => {
      clearInterval(interval);
      placeholder.style.display = 'none';
      if (placeholder.parentElement !== document.body) {
        document.body.appendChild(placeholder);
      }
    };
  }, []);

  return <div ref={containerRef} />;
};

export default GoogleTranslate;
