export default function StepsSection() {
  const steps = [
    {
      id: 1,
      number: '01',
      title: 'Capture',
      description: 'Take a photo of the affected plant leaf using your camera or upload from gallery',
      iconBg: 'bg-gradient-to-br from-blue-500 to-blue-600',
      iconShadow: 'shadow-blue-500/50',
      bgColor: 'bg-gradient-to-br from-blue-50 to-blue-100',
      borderColor: 'border-blue-300',
      badgeColor: 'bg-blue-500 text-white',
      glowColor: 'bg-blue-400',
      icon: (
        <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      ),
    },
    {
      id: 2,
      number: '02',
      title: 'Analyze',
      description: 'Our AI analyzes the image using advanced machine learning algorithms in seconds',
      iconBg: 'bg-gradient-to-br from-purple-500 to-purple-600',
      iconShadow: 'shadow-purple-500/50',
      bgColor: 'bg-gradient-to-br from-purple-50 to-purple-100',
      borderColor: 'border-purple-300',
      badgeColor: 'bg-purple-500 text-white',
      glowColor: 'bg-purple-400',
      icon: (
        <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
      ),
    },
    {
      id: 3,
      number: '03',
      title: 'Treat',
      description: 'Get instant diagnosis and personalized treatment recommendations for your crops',
      iconBg: 'bg-gradient-to-br from-green-500 to-green-600',
      iconShadow: 'shadow-green-500/50',
      bgColor: 'bg-gradient-to-br from-green-50 to-green-100',
      borderColor: 'border-green-300',
      badgeColor: 'bg-green-500 text-white',
      glowColor: 'bg-green-400',
      icon: (
        <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
      ),
    },
  ];

  return (
    <section id="how-it-works" className="py-24 bg-white -mt-1 relative overflow-hidden">
      {/* Background Decorative Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-10 w-72 h-72 bg-blue-100 rounded-full opacity-20 blur-3xl" />
        <div className="absolute bottom-20 right-10 w-96 h-96 bg-purple-100 rounded-full opacity-20 blur-3xl" />
      </div>

      <div className="container mx-auto px-6 relative z-10">
        {/* Section Header */}
        <div className="text-center mb-20">
          <div className="inline-flex items-center space-x-2 bg-agri-green bg-opacity-10 px-4 py-2 rounded-full mb-6 cursor-pointer hover:scale-105 transition-transform">
            <span className="text-agri-green font-semibold text-sm">HOW IT WORKS</span>
          </div>

          <h2 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6 hover:scale-102 transition-transform">
            Three Simple Steps to{' '}
            <span className="bg-gradient-to-r from-green-600 via-emerald-500 to-green-600 bg-clip-text text-transparent">
              Healthier Crops
            </span>
          </h2>

          <p className="text-gray-600 text-lg max-w-2xl mx-auto">
            Revolutionary farming with AI & ML. Get instant crop solutions in just three easy steps.
          </p>
        </div>

        {/* Steps Grid */}
        <div className="grid md:grid-cols-3 gap-10">
          {steps.map((step, index) => (
            <div key={step.id} className="relative group">
              {/* Connecting Line (not on last item) */}
              {index < steps.length - 1 && (
                <div className="hidden md:block absolute top-1/4 left-full w-10 h-0.5 bg-gradient-to-r from-gray-300 to-transparent z-0" />
              )}

              {/* Card */}
              <div className={`relative ${step.bgColor} p-10 rounded-3xl border-2 ${step.borderColor} transition-all duration-300 shadow-xl hover:shadow-2xl cursor-pointer h-full overflow-hidden hover:-translate-y-4 hover:scale-105`}>
                {/* Animated gradient overlay on hover */}
                <div className="absolute inset-0 bg-gradient-to-br from-white to-transparent opacity-0 group-hover:opacity-50 transition-opacity duration-300" />

                {/* Step Number Badge */}
                <div className={`absolute top-6 right-6 w-14 h-14 rounded-full ${step.badgeColor} flex items-center justify-center font-bold text-lg shadow-lg group-hover:rotate-360 group-hover:scale-110 transition-all duration-600`}>
                  {step.number}
                </div>

                <div className="relative z-10">
                  {/* Icon Container */}
                  <div className="relative mb-8 w-24 h-24">
                    {/* Icon background */}
                    <div className={`relative ${step.iconBg} ${step.iconShadow} w-full h-full rounded-2xl flex items-center justify-center shadow-2xl group-hover:scale-110 group-hover:rotate-5 transition-all duration-300`}>
                      {step.icon}
                    </div>

                    {/* Glow effect */}
                    <div className={`absolute inset-0 ${step.glowColor} rounded-2xl opacity-0 group-hover:opacity-20 blur-xl transition-opacity duration-300`} />
                  </div>

                  {/* Title */}
                  <h3 className="text-3xl font-bold text-gray-900 mb-4">
                    {step.title}
                  </h3>

                  {/* Description */}
                  <p className="text-gray-700 leading-relaxed text-base">
                    {step.description}
                  </p>

                  {/* Decorative line */}
                  <div className={`mt-6 h-1 ${step.iconBg} rounded-full w-full`} />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
