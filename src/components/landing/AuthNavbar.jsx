import { useNavigate } from 'react-router-dom';

export default function AuthNavbar() {
    const navigate = useNavigate();

    return (
        <nav className="fixed w-full top-0 z-50 bg-transparent backdrop-blur-md bg-opacity-30">
            <div className="container mx-auto px-6 h-[72px] flex items-center justify-between">
                {/* Logo Only */}
                <button
                    onClick={() => navigate('/')}
                    className="text-2xl font-semibold text-agri-green hover:text-agri-green-light transition-colors duration-300 cursor-pointer"
                >
                    AgriDetect AI
                </button>
            </div>
        </nav>
    );
}

