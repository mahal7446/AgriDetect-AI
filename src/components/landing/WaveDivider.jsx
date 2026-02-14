export default function WaveDivider() {
  return (
    <div className="absolute bottom-0 left-0 right-0 w-full overflow-hidden leading-none">
      <svg
        className="block w-full h-auto"
        viewBox="0 0 1200 120"
        preserveAspectRatio="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Wave path from green to white */}
        <path
          d="M0,40 Q300,20 600,40 T1200,40 L1200,120 L0,120 Z"
          fill="white"
          opacity="1"
        />
        
        {/* Subtle second wave for depth */}
        <path
          d="M0,50 Q300,30 600,50 T1200,50 L1200,120 L0,120 Z"
          fill="white"
          opacity="0.05"
        />
      </svg>
    </div>
  );
}
