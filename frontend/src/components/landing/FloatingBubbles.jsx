import React from 'react';

const FloatingBubbles = () => {
  // Generate random bubbles on component mount
  const bubbles = React.useMemo(() => {
    return Array.from({ length: 8 }).map((_, i) => ({
      id: i,
      size: Math.random() * 150 + 50, // 50-200px
      left: Math.random() * 100, // 0-100%
      delay: Math.random() * 5, // 0-5s
      duration: Math.random() * 15 + 20, // 20-35s
      opacity: Math.random() * 0.3 + 0.05, // 0.05-0.35
    }));
  }, []);

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none z-0">
      <style>{`
        @keyframes float-up {
          0% {
            transform: translateY(100vh) translateX(0);
            opacity: 0;
          }
          10% {
            opacity: var(--bubble-opacity);
          }
          90% {
            opacity: var(--bubble-opacity);
          }
          100% {
            transform: translateY(-100vh) translateX(100px);
            opacity: 0;
          }
        }
        
        .floating-bubble {
          position: absolute;
          border-radius: 50%;
          filter: blur(2px);
        }
      `}</style>

      {bubbles.map((bubble) => (
        <div
          key={bubble.id}
          className="floating-bubble"
          style={{
            width: `${bubble.size}px`,
            height: `${bubble.size}px`,
            left: `${bubble.left}%`,
            bottom: `-${bubble.size}px`,
            background: bubble.id % 2 === 0 
              ? `rgba(255, 255, 255, ${bubble.opacity})` 
              : `rgba(34, 197, 94, ${bubble.opacity * 0.8})`,
            animation: `float-up ${bubble.duration}s infinite linear`,
            animationDelay: `-${bubble.delay}s`,
            '--bubble-opacity': bubble.opacity,
          }}
        />
      ))}
    </div>
  );
};

export default FloatingBubbles;
