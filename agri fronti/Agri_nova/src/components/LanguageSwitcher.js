import React from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { Globe } from 'lucide-react';
import './LanguageSwitcher.css';

const LanguageSwitcher = () => {
  const { currentLanguage, changeLanguage, availableLanguages } = useLanguage();
  const [isOpen, setIsOpen] = React.useState(false);

  return (
    <div className="language-switcher-container">
      <button 
        className="language-switcher-button"
        onClick={() => setIsOpen(!isOpen)}
        title="Change Language"
      >
        <Globe className="w-5 h-5" />
        <span className="language-code">{currentLanguage.toUpperCase()}</span>
      </button>

      {isOpen && (
        <div className="language-dropdown">
          {availableLanguages.map((lang) => (
            <button
              key={lang.code}
              onClick={() => {
                changeLanguage(lang.code);
                setIsOpen(false);
              }}
              className={`language-option ${currentLanguage === lang.code ? 'active' : ''}`}
            >
              <span className="language-native">{lang.nativeName}</span>
              <span className="language-english">({lang.name})</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default LanguageSwitcher;
