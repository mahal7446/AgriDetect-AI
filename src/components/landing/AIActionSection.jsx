import React from 'react';

export default function AIActionSection() {
  const featureGroups = [
    {
      category: 'Diseases',
      items: ['Maize rust', 'Leaf blight', 'Fungal infections', 'Bacterial diseases'],
    },
    {
      category: 'Pest Damage',
      items: ['Fall armyworm', 'Stem borer', 'Aphids', 'Destructive insects'],
    },
    {
      category: 'Nutrient Deficiencies',
      items: ['Nitrogen', 'Phosphorus', 'Potassium', 'Micronutrients'],
    },
    {
      category: 'Abiotic Stress',
      items: ['Drought stress', 'Cold damage', 'Heat stress', 'Environmental factors'],
    },
  ];

  return (
    <section id="services" className="py-24 bg-agri-gradient">
      <style>{`
        @keyframes float-marker {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-8px); }
        }
        
        @keyframes subtle-pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.8; transform: scale(1.1); }
        }
        
        @keyframes fade-in-up {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes shimmer {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        
        .fade-in-up {
          animation: fade-in-up 0.6s ease-out forwards;
        }
        
        .float-marker {
          animation: float-marker 3s ease-in-out infinite;
        }
        
        .subtle-pulse {
          animation: subtle-pulse 2s ease-in-out infinite;
        }

        .shimmer-text {
          background-size: 200% auto;
          animation: shimmer 4s linear infinite;
        }
      `}</style>

      <div className="container mx-auto px-6">
        {/* Section Header */}
        <div className="max-w-3xl mb-16 animate-in fade-in slide-in-from-bottom-4 duration-600">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-6 hover:scale-102 transition-transform">
            See AI in{' '}
            <span className="bg-gradient-to-r from-green-400 via-emerald-300 to-green-400 bg-clip-text text-transparent shimmer-text">
              Action
            </span>
          </h2>
          <p className="text-green-100 text-lg leading-relaxed font-medium">
            Our advanced computer vision technology powered by deep learning can identify early signs of crop problems before they spread.
          </p>
        </div>

        {/* Two Column Layout */}
        <div className="grid lg:grid-cols-2 gap-12 items-stretch">
          {/* Left: Feature Cards - 2x2 Grid */}
          <div className="grid grid-cols-2 gap-5">
            {featureGroups.map((group, idx) => (
              <div
                key={idx}
                style={{ animationDelay: `${idx * 100}ms` }}
                className="h-48 bg-gradient-to-br from-green-900/30 to-green-950/40 backdrop-blur-md rounded-xl p-6 border-2 border-agri-green-light/60 flex flex-col shadow-md cursor-pointer relative overflow-hidden group hover:-translate-y-2 hover:scale-105 hover:border-green-400 hover:shadow-2xl hover:shadow-green-400/30 transition-all duration-300 animate-in fade-in zoom-in-95"
              >
                {/* Shine effect on hover */}
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-600"></div>

                <h3 className="text-white font-bold text-base mb-3 flex items-center relative z-10">
                  <span className="w-5 h-5 rounded-full bg-agri-green text-agri-button-dark flex items-center justify-center text-xs font-bold mr-2 hover:rotate-360 hover:scale-120 transition-all duration-500">
                    ✓
                  </span>
                  {group.category}
                </h3>
                <ul className="space-y-1.5 flex-1 overflow-hidden relative z-10">
                  {group.items.map((item, i) => (
                    <li
                      key={i}
                      className="text-green-100 text-xs ml-7 flex items-start font-medium cursor-pointer hover:translate-x-1 hover:text-green-400 transition-all"
                    >
                      <span className="w-1 h-1 bg-agri-green rounded-full mr-2 mt-1.5 flex-shrink-0"></span>
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>

          {/* Right: Image with Detection Markers */}
          <div className="relative animate-in fade-in slide-in-from-right duration-700" style={{ animationDelay: '400ms' }}>
            {/* Image Container */}
            <div className="relative w-full h-[500px] rounded-3xl overflow-hidden shadow-2xl group hover:scale-102 transition-transform duration-400">
              {/* Background Image with Gradient Overlay */}
              <div
                className="absolute inset-0 bg-cover bg-center transition-all duration-500 group-hover:scale-105 group-hover:brightness-110"
                style={{
                  backgroundImage: 'url("/corn.jpg")',
                }}
              ></div>

              {/* Overlay gradient */}
              <div className="absolute inset-0 bg-gradient-to-t from-black/30 via-transparent to-black/20"></div>

              {/* Detection Markers */}
              {/* Red Marker - Leaf Spot */}
              <div className="absolute top-1/3 left-1/4 float-marker">
                <div className="relative">
                  <div className="w-12 h-12 bg-red-500 rounded-full border-4 border-red-300 shadow-2xl shadow-red-500/60 flex items-center justify-center cursor-pointer subtle-pulse hover:scale-130 transition-transform">
                    <span className="text-white text-xl">⚠</span>
                  </div>
                  <div className="absolute top-16 left-1/2 transform -translate-x-1/2 bg-red-600 text-white px-5 py-2.5 rounded-full text-sm font-bold whitespace-nowrap shadow-xl shadow-red-600/60 border-2 border-red-400 hover:scale-105 hover:-translate-y-0.5 transition-all">
                    Leaf Spot Detected
                  </div>
                </div>
              </div>

              {/* Yellow Marker - Nutrient Deficiency */}
              <div className="absolute bottom-1/3 right-1/4 float-marker" style={{ animationDelay: '1s' }}>
                <div className="relative">
                  <div className="w-12 h-12 bg-yellow-400 rounded-full border-4 border-yellow-200 shadow-2xl shadow-yellow-400/60 flex items-center justify-center cursor-pointer subtle-pulse hover:scale-130 transition-transform" style={{ animationDelay: '0.5s' }}>
                    <span className="text-yellow-900 text-xl">⚡</span>
                  </div>
                  <div className="absolute top-16 left-1/2 transform -translate-x-1/2 bg-amber-600 text-white px-5 py-2.5 rounded-full text-sm font-bold whitespace-nowrap shadow-xl shadow-amber-600/60 border-2 border-amber-400 hover:scale-105 hover:-translate-y-0.5 transition-all">
                    Nutrient Deficiency
                  </div>
                </div>
              </div>

              {/* Bottom Analysis Card */}
              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-gray-950 via-gray-900 to-transparent p-8 backdrop-blur-sm">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-white font-bold text-lg">Analysis Complete</span>
                    <span className="bg-agri-green text-agri-button-dark px-4 py-2 rounded-full text-sm font-bold border border-agri-green-light shadow-lg cursor-pointer hover:scale-110 transition-transform">
                      98% Confidence
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2.5 overflow-hidden border border-gray-600 shadow-md">
                    <div className="bg-gradient-to-r from-agri-green to-agri-green-light h-2.5 rounded-full shadow-lg shadow-agri-green/50 animate-in slide-in-from-left duration-1500" style={{ width: '98%', animationDelay: '1s' }}></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
