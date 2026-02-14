import { useNavigate } from 'react-router-dom';
import Navbar from '@/components/landing/Navbar';
import Hero from '@/components/landing/Hero';
import StepsSection from '@/components/landing/StepsSection';
import AIActionSection from '@/components/landing/AIActionSection';
import Features from '@/components/landing/Features';
import CTA from '@/components/landing/CTA';
import Footer from '@/components/landing/Footer';

export const LandingPage = () => {
  const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate('/login');
  };

  return (
    <div className="w-full">
      <Navbar onGetStarted={handleGetStarted} />
      <Hero onGetStarted={handleGetStarted} />
      <StepsSection />
      <AIActionSection />
      <Features />
      <CTA onGetStarted={handleGetStarted} />
      <Footer />
    </div>
  );
};
