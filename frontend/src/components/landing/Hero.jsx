import FloatingBubbles from './FloatingBubbles';
import WaveDivider from './WaveDivider';

export default function Hero({ onGetStarted }) {
  const handleScroll = (id) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section id="hero" className="relative min-h-screen bg-agri-gradient flex items-center pt-[72px] overflow-visible pb-20">
      <FloatingBubbles />
      <div className="container mx-auto px-6 w-full relative z-10">
        <div className="max-w-3xl">
          {/* Badge */}
          <div className="inline-flex items-center space-x-2 bg-green-900 bg-opacity-40 px-4 py-2 rounded-full mb-8 border border-agri-green-light border-opacity-30">
            <svg
              className="w-4 h-4 text-agri-green"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path d="M11 3a1 1 0 10-2 0v1a1 1 0 102 0V3zM15.657 5.757a1 1 0 00-1.414-1.414l-.707.707a1 1 0 001.414 1.414l.707-.707zM18 10a1 1 0 01-1 1h-1a1 1 0 110-2h1a1 1 0 011 1zM15.657 14.243a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414l.707.707zM11 17a1 1 0 102 0v-1a1 1 0 10-2 0v1zM5.757 15.657a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414l-.707.707zM2 10a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.757 4.343a1 1 0 011.414 1.414l-.707.707a1 1 0 01-1.414-1.414l.707-.707z" />
            </svg>
            <span className="text-sm font-medium text-agri-muted">
              AI-Powered Plant Disease Detection
            </span>
          </div>

          {/* Headline */}
          <h1 className="text-5xl md:text-6xl font-bold leading-tight mb-6 text-white">
            Cultivating the<br />
            Future with{' '}
            <span className="text-agri-green">AI-Powered</span>
            <br />
            <span className="text-agri-green">Agriculture</span>
          </h1>

          {/* Description */}
          <p className="text-lg text-agri-muted mb-8 max-w-2xl leading-relaxed">
            Fast, free, and smart crop problem solver — just snap a pic
          </p>

          {/* Buttons */}
          <div className="flex flex-col sm:flex-row gap-4">
            <button
              onClick={onGetStarted}
              className="px-8 py-3 bg-agri-green text-agri-button-dark font-semibold rounded-full hover:bg-agri-green-light transition-all duration-300 shadow-lg hover:shadow-xl"
            >
              Getting Started
            </button>
            <button
              onClick={() => handleScroll('how-it-works')}
              className="px-8 py-3 border-2 border-agri-green-light text-white font-semibold rounded-full hover:bg-agri-green-light hover:text-agri-button-dark transition-all duration-300"
            >
              See How It Works
            </button>
          </div>
        </div>
      </div>

      {/* Wave Divider */}
      <WaveDivider />
    </section>
  );
}

