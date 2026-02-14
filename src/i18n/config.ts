import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

// Import translation files
import en from './locales/en.json';
import hi from './locales/hi.json';
import kn from './locales/kn.json';
import te from './locales/te.json';
import ta from './locales/ta.json';
import bn from './locales/bn.json';

const resources = {
    en: { translation: en },
    hi: { translation: hi },
    kn: { translation: kn },
    te: { translation: te },
    ta: { translation: ta },
    bn: { translation: bn },
};

i18n
    .use(LanguageDetector) // Detect user language
    .use(initReactI18next) // Pass i18n instance to react-i18next
    .init({
        resources,
        fallbackLng: 'en', // Fallback language if current language is not available
        lng: localStorage.getItem('language') || 'en', // Default language

        interpolation: {
            escapeValue: false, // React already escapes values
        },

        detection: {
            // Order of language detection methods
            order: ['localStorage', 'navigator'],
            caches: ['localStorage'],
        },
    });

export default i18n;
