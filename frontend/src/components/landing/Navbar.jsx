import { useState, useEffect } from 'react';

export default function Navbar({ onGetStarted }) {
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScrollEvent = () => {
      setIsScrolled(window.scrollY > 50);
    };

    window.addEventListener('scroll', handleScrollEvent);
    return () => window.removeEventListener('scroll', handleScrollEvent);
  }, []);

  const handleScroll = (id) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <nav className={`fixed w-full top-0 z-50 transition-all duration-500 ${isScrolled
        ? 'bg-white/30 backdrop-blur-xl border-b border-white/20 shadow-lg py-3'
        : 'bg-transparent py-4'
      }`}>
      <div className="container mx-auto px-6 flex items-center justify-between">
        {/* Logo */}
        <div className={`text-2xl font-bold transition-colors duration-300 ${isScrolled ? 'text-agri-button-dark' : 'text-white'
          }`}>
          AgriDetect AI
        </div>

        {/* Nav Links - Hidden on mobile */}
        <div className="hidden md:flex items-center space-x-8">
          {['about', 'services', 'features', 'how-it-works'].map((item) => (
            <button
              key={item}
              onClick={() => handleScroll(item === 'about' ? 'hero' : item)}
              className={`font-medium transition-all duration-300 hover:text-agri-green ${isScrolled ? 'text-gray-700' : 'text-white'
                }`}
            >
              {item.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
            </button>
          ))}
        </div>

        {/* CTA Button */}
        <button
          onClick={onGetStarted}
          className={`font-semibold px-6 py-2.5 rounded-full transition-all duration-300 shadow-lg hover:shadow-xl ${isScrolled
            ? 'bg-agri-green text-white hover:bg-agri-green-light'
            : 'bg-white text-agri-button-dark hover:bg-agri-green-light'
            }`}
        >
          Get Started
        </button>
      </div>
    </nav>
  );
}
