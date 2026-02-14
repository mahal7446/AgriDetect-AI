import { useLocation, Link } from "react-router-dom";
import { useEffect } from "react";
import { Leaf, Home, ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname
    );
  }, [location.pathname]);

  return (
    <div className="min-h-screen flex items-center justify-center p-4 gradient-earth">
      <div className="text-center space-y-6 animate-fade-in">
        <div className="mx-auto p-6 rounded-full bg-primary/10 w-fit">
          <Leaf className="w-16 h-16 text-primary" />
        </div>
        
        <div className="space-y-2">
          <h1 className="text-6xl font-bold text-primary">404</h1>
          <h2 className="text-2xl font-semibold text-foreground">Page Not Found</h2>
          <p className="text-muted-foreground max-w-md mx-auto">
            Oops! It looks like this page has wilted away. 
            Let's get you back to the fields.
          </p>
        </div>

        <div className="flex flex-col sm:flex-row gap-3 justify-center">
          <Button asChild className="gap-2">
            <Link to="/dashboard">
              <Home className="w-4 h-4" />
              Go to Dashboard
            </Link>
          </Button>
          <Button variant="outline" asChild className="gap-2">
            <Link to="/login">
              <ArrowLeft className="w-4 h-4" />
              Back to Login
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
};

export default NotFound;
