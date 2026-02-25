import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Leaf, Mail, Lock, Eye, EyeOff, Loader2, User, Phone, Check, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import AuthNavbar from "@/components/landing/AuthNavbar";
import { API_BASE_URL } from "@/lib/api";

export const SignupPage = () => {
  const [formData, setFormData] = useState({
    fullName: "",
    email: "",
    phone: "",
    password: "",
    confirmPassword: "",
  });
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();
  const { toast } = useToast();

  // Password validation state
  const [passwordValidation, setPasswordValidation] = useState({
    length: false,
    uppercase: false,
    lowercase: false,
    number: false,
    special: false,
  });

  // Phone validation state
  const [phoneValid, setPhoneValid] = useState<boolean | null>(null);

  const validatePasswordRules = (password: string) => {
    setPasswordValidation({
      length: password.length >= 8 && password.length <= 12,
      uppercase: /[A-Z]/.test(password),
      lowercase: /[a-z]/.test(password),
      number: /[0-9]/.test(password),
      special: /[!@#$%^&*(),.?":{}|<>_\-+=\[\]\\;/`~]/.test(password),
    });
  };

  const validatePhoneNumber = (phone: string) => {
    const cleaned = phone.replace(/[\s\-().]/g, '');
    if (cleaned.length === 0) {
      setPhoneValid(null);
    } else {
      setPhoneValid(/^\d{10}$/.test(cleaned));
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });

    // Real-time validation
    if (name === 'password') {
      validatePasswordRules(value);
    } else if (name === 'phone') {
      validatePhoneNumber(value);
    }
  };

  const isPasswordValid = () => {
    return Object.values(passwordValidation).every(v => v === true);
  };

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();

    // Client-side validation
    if (!isPasswordValid()) {
      toast({
        title: "Invalid Password",
        description: "Please ensure your password meets all requirements",
        variant: "destructive",
      });
      return;
    }

    if (phoneValid !== true) {
      toast({
        title: "Invalid Phone Number",
        description: "Please enter a valid 10-digit phone number",
        variant: "destructive",
      });
      return;
    }

    if (formData.password !== formData.confirmPassword) {
      toast({
        title: "Passwords don't match",
        description: "Please make sure your passwords match",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/signup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (data.success) {
        toast({
          title: "Account created successfully! 🎉",
          description: "You can now log in with your credentials",
        });

        // Redirect to login page instead of email verification
        navigate("/login");
      } else {
        toast({
          title: "Signup failed",
          description: data.error || "An error occurred",
          variant: "destructive",
        });
      }
    } catch (error) {
      toast({
        title: "Network error",
        description: "Could not connect to server",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      {/* Simplified Auth Navbar */}
      <AuthNavbar />

      <div className="min-h-screen flex items-center justify-center p-4 gradient-earth pt-20">
        {/* Decorative Elements */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute -top-20 -left-20 w-64 h-64 bg-primary/5 rounded-full blur-3xl" />
          <div className="absolute -bottom-20 -right-20 w-80 h-80 bg-harvest/10 rounded-full blur-3xl" />
        </div>

        <Card className="w-full max-w-md relative z-10 shadow-2xl border-border/50 animate-fade-in">
          <CardHeader className="text-center pb-2">
            <div className="mx-auto mb-4 p-4 rounded-2xl bg-primary/10 w-fit">
              <Leaf className="w-12 h-12 text-primary" />
            </div>
            <CardTitle className="text-2xl font-bold">
              Join <span className="text-primary">AgriDetect AI</span>
            </CardTitle>
            <CardDescription className="text-base">
              Create your account and start protecting your crops
            </CardDescription>
          </CardHeader>

          <form onSubmit={handleSignup}>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="fullName">Full Name</Label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                  <Input
                    id="fullName"
                    name="fullName"
                    type="text"
                    placeholder="Enter your full name"
                    className="pl-10"
                    value={formData.fullName}
                    onChange={handleChange}
                    required
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="email">Email Address</Label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                  <Input
                    id="email"
                    name="email"
                    type="email"
                    placeholder="farmer@example.com"
                    className="pl-10"
                    value={formData.email}
                    onChange={handleChange}
                    required
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="phone">Phone Number *</Label>
                <div className="relative">
                  <Phone className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                  <Input
                    id="phone"
                    name="phone"
                    type="tel"
                    placeholder="9876543210"
                    className={`pl-10 pr-10 ${phoneValid === true ? 'border-green-500' : phoneValid === false ? 'border-red-500' : ''}`}
                    value={formData.phone}
                    onChange={handleChange}
                    required
                  />
                  {phoneValid !== null && (
                    <div className="absolute right-3 top-1/2 -translate-y-1/2">
                      {phoneValid ? (
                        <Check className="w-5 h-5 text-green-500" />
                      ) : (
                        <X className="w-5 h-5 text-red-500" />
                      )}
                    </div>
                  )}
                </div>
                {phoneValid === false && (
                  <p className="text-xs text-red-500">Phone number must be exactly 10 digits</p>
                )}
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-2">
                  <Label htmlFor="password">Password *</Label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                    <Input
                      id="password"
                      name="password"
                      type={showPassword ? "text" : "password"}
                      placeholder="••••••••"
                      className="pl-9 pr-9"
                      value={formData.password}
                      onChange={handleChange}
                      required
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                    >
                      {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="confirmPassword">Confirm *</Label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                    <Input
                      id="confirmPassword"
                      name="confirmPassword"
                      type={showPassword ? "text" : "password"}
                      placeholder="••••••••"
                      className="pl-9"
                      value={formData.confirmPassword}
                      onChange={handleChange}
                      required
                    />
                  </div>
                </div>
              </div>

              {/* Password Requirements */}
              {formData.password && (
                <div className="bg-muted/50 border border-border rounded-lg p-3 space-y-2">
                  <p className="text-xs font-semibold text-muted-foreground">Password must contain:</p>
                  <div className="grid grid-cols-1 gap-1 text-xs">
                    <div className={`flex items-center gap-2 ${passwordValidation.length ? 'text-green-600' : 'text-muted-foreground'}`}>
                      {passwordValidation.length ? <Check className="w-3 h-3" /> : <X className="w-3 h-3" />}
                      <span>8-12 characters</span>
                    </div>
                    <div className={`flex items-center gap-2 ${passwordValidation.uppercase ? 'text-green-600' : 'text-muted-foreground'}`}>
                      {passwordValidation.uppercase ? <Check className="w-3 h-3" /> : <X className="w-3 h-3" />}
                      <span>Uppercase letter (A-Z)</span>
                    </div>
                    <div className={`flex items-center gap-2 ${passwordValidation.lowercase ? 'text-green-600' : 'text-muted-foreground'}`}>
                      {passwordValidation.lowercase ? <Check className="w-3 h-3" /> : <X className="w-3 h-3" />}
                      <span>Lowercase letter (a-z)</span>
                    </div>
                    <div className={`flex items-center gap-2 ${passwordValidation.number ? 'text-green-600' : 'text-muted-foreground'}`}>
                      {passwordValidation.number ? <Check className="w-3 h-3" /> : <X className="w-3 h-3" />}
                      <span>Number (0-9)</span>
                    </div>
                    <div className={`flex items-center gap-2 ${passwordValidation.special ? 'text-green-600' : 'text-muted-foreground'}`}>
                      {passwordValidation.special ? <Check className="w-3 h-3" /> : <X className="w-3 h-3" />}
                      <span>Special character (!, @, #, $, etc.)</span>
                    </div>
                  </div>
                </div>
              )}

              <label className="flex items-center gap-2 cursor-pointer text-sm">
                <input type="checkbox" className="rounded border-border" required />
                <span className="text-muted-foreground">
                  I agree to the{" "}
                  <Link to="/terms" className="text-primary hover:underline">Terms of Service</Link>
                  {" "}and{" "}
                  <Link to="/privacy" className="text-primary hover:underline">Privacy Policy</Link>
                </span>
              </label>
            </CardContent>

            <CardFooter className="flex flex-col gap-4">
              <Button
                type="submit"
                className="w-full h-12 text-base font-semibold shadow-glow"
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    Creating account...
                  </>
                ) : (
                  "Create Account"
                )}
              </Button>

              <p className="text-center text-muted-foreground">
                Already have an account?{" "}
                <Link to="/login" className="text-primary font-semibold hover:underline">
                  Sign in
                </Link>
              </p>
            </CardFooter>
          </form>
        </Card>
      </div>
    </>
  );
};
