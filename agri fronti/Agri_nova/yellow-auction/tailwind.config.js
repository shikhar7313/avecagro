/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './public/index.html',
    './src/**/*.{js,jsx,ts,tsx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular'],
      },
      colors: {
        brand: {
          yellow: {
            50: '#fefce8',
            100: '#fef9c3',
            200: '#fef08a',
            300: '#fde047',
            400: '#facc15',
            500: '#eab308',
            600: '#ca8a04',
            700: '#a16207',
            800: '#854d0e',
            900: '#713f12',
          },
          amber: {
            400: '#f59e0b',
            500: '#d97706',
            600: '#b45309',
          },
          dark: '#0a0a0a',
          accent: '#ffea00'
        }
      },
      boxShadow: {
        'brand-glow': '0 0 25px -5px rgba(250,204,21,0.6)',
        'brand-inner': 'inset 0 1px 4px 0 rgba(255,255,255,0.15), 0 4px 12px -3px rgba(0,0,0,0.4)'
      },
      animation: {
        'spin-slow': 'spin 20s linear infinite',
        'pulse-fast': 'pulse 1s ease-in-out infinite',
        'fade-in': 'fade-in 0.5s both',
        'slide-up': 'slide-up 0.6s both',
        'gradient-x': 'gradient-x 6s ease infinite',
      },
      keyframes: {
        'fade-in': {
          '0%': { opacity: 0, transform: 'translateY(20px)' },
          '100%': { opacity: 1, transform: 'translateY(0)' }
        },
        'slide-up': {
          '0%': { opacity: 0, transform: 'translateY(40px)' },
          '100%': { opacity: 1, transform: 'translateY(0)' }
        },
        'gradient-x': {
          '0%, 100%': { 'background-position': '0% 50%' },
          '50%': { 'background-position': '100% 50%' }
        }
      }
    },
  },
  plugins: [
    function ({ addUtilities }) {
      const newUtils = {
        '.text-gradient-brand': {
          background: 'linear-gradient(90deg, #facc15, #f59e0b, #d97706)',
          '-webkit-background-clip': 'text',
          'background-clip': 'text',
          color: 'transparent'
        },
        '.bg-grid-yellow': {
          background: 'linear-gradient(rgba(250,204,21,0.15) 1px, transparent 1px), linear-gradient(90deg, rgba(250,204,21,0.15) 1px, transparent 1px)',
          'background-size': '40px 40px'
        }
      }
      addUtilities(newUtils, ['responsive'])
    }
  ],
}

