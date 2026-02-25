export default function Features() {
  const features = [
    {
      id: 'detection',
      icon: (
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
        </svg>
      ),
      title: 'Instant Detection',
      description: 'Identify plant diseases in seconds with advanced AI algorithms',
      color: 'from-blue-500 to-cyan-500',
      lightColor: 'bg-blue-50',
    },
    {
      id: 'analysis',
      icon: (
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      ),
      title: 'Detailed Analysis',
      description: 'Get comprehensive insights into disease severity and progression',
      color: 'from-purple-500 to-indigo-500',
      lightColor: 'bg-purple-50',
    },
    {
      id: 'solutions',
      icon: (
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      ),
      title: 'Smart Solutions',
      description: 'Receive personalized treatment recommendations for your crops',
      color: 'from-amber-500 to-orange-500',
      lightColor: 'bg-amber-50',
    },
    {
      id: 'tracking',
      icon: (
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
        </svg>
      ),
      title: 'Progress Tracking',
      description: 'Monitor your crop health over time with detailed reports',
      color: 'from-green-500 to-emerald-500',
      lightColor: 'bg-green-50',
    },
  ];

  return (
    <section id="features" className="py-28 bg-white relative overflow-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-agri-green/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />
        <div className="absolute bottom-0 left-0 w-[400px] h-[400px] bg-blue-500/5 rounded-full blur-3xl translate-y-1/2 -translate-x-1/2" />
      </div>

      <div className="container mx-auto px-6 relative z-10">
        {/* Section Header */}
        <div className="text-center mb-20">
          <div className="inline-block px-4 py-1.5 mb-6 text-sm font-semibold tracking-wider text-agri-green uppercase bg-agri-green/10 rounded-full">
            Capabilities
          </div>
          <h2 className="text-5xl md:text-6xl font-bold mb-6 tracking-tight text-gray-900">
            Our <span className="bg-gradient-to-r from-agri-green via-emerald-500 to-agri-green-light bg-clip-text text-transparent animate-gradient">Features</span>
          </h2>
          <p className="text-gray-600 text-lg max-w-2xl mx-auto leading-relaxed">
            Empowering farmers with cutting-edge AI technology. Everything you need to keep your crops healthy and productive in one smart platform.
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature) => (
            <div
              key={feature.id}
              className="group relative bg-white p-10 rounded-[2.5rem] border border-gray-100 shadow-xl shadow-gray-200/50 hover:shadow-2xl hover:shadow-agri-green/20 transition-all duration-500 hover:-translate-y-2"
            >
              {/* Card Hover Effect Background */}
              <div className="absolute inset-0 bg-gradient-to-br from-transparent to-gray-50/50 rounded-[2.5rem] opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

              <div className="relative z-10">
                {/* Icon Container */}
                <div className={`w-16 h-16 mb-8 rounded-2xl bg-gradient-to-br ${feature.color} flex items-center justify-center text-white shadow-lg group-hover:scale-110 group-hover:rotate-6 transition-all duration-500`}>
                  {feature.icon}
                </div>

                <h3 className="text-2xl font-bold text-gray-900 mb-4 group-hover:text-agri-green transition-colors duration-300">
                  {feature.title}
                </h3>

                <p className="text-gray-600 leading-relaxed text-base">
                  {feature.description}
                </p>

                {/* Decorative Bottom Bar */}
                <div className="mt-8 h-1 w-0 bg-gradient-to-r from-transparent via-agri-green to-transparent group-hover:w-full transition-all duration-700 mx-auto rounded-full" />
              </div>

              {/* Top-right Accent Icon/Shape */}
              <div className="absolute top-6 right-6 opacity-0 group-hover:opacity-10 transition-opacity duration-500 rotate-12">
                <div className={`w-12 h-12 rounded-full bg-gradient-to-br ${feature.color}`} />
              </div>
            </div>
          ))}
        </div>
      </div>

      <style jsx>{`
        @keyframes gradient {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        .animate-gradient {
          background-size: 200% auto;
          animation: gradient 3s linear infinite;
        }
      `}</style>
    </section>
  );
}
