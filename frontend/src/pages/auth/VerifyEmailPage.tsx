import { useState, useEffect } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { Loader2, Mail, CheckCircle2, XCircle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import AuthNavbar from "@/components/landing/AuthNavbar";
import { API_BASE_URL } from "@/lib/api";

export const VerifyEmailPage = () => {
    const { token } = useParams();
    const navigate = useNavigate();
    const { toast } = useToast();

    const [status, setStatus] = useState<'verifying' | 'success' | 'error' | 'pending'>('pending');
    const [message, setMessage] = useState("");
    const [email, setEmail] = useState("");
    const [isResending, setIsResending] = useState(false);

    useEffect(() => {
        if (token) {
            verifyEmail(token);
        }
    }, [token]);

    const verifyEmail = async (verificationToken: string) => {
        setStatus('verifying');

        try {
            const response = await fetch(`${API_BASE_URL}/api/auth/verify-email/${verificationToken}`);
            const data = await response.json();

            if (data.success) {
                setStatus('success');
                setMessage(data.message);

                toast({
                    title: "Email Verified! 🎉",
                    description: "Your account has been activated successfully",
                });

                // Redirect to login after 3 seconds
                setTimeout(() => {
                    navigate('/login');
                }, 3000);
            } else {
                setStatus('error');
                setMessage(data.error || "Verification failed");

                toast({
                    title: "Verification Failed",
                    description: data.error,
                    variant: "destructive",
                });
            }
        } catch (error) {
            setStatus('error');
            setMessage("Network error. Please try again.");

            toast({
                title: "Error",
                description: "Could not connect to server",
                variant: "destructive",
            });
        }
    };

    const resendVerification = async () => {
        if (!email) {
            toast({
                title: "Email Required",
                description: "Please enter your email address",
                variant: "destructive",
            });
            return;
        }

        setIsResending(true);

        try {
            const response = await fetch(`${API_BASE_URL}/api/auth/resend-verification`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email }),
            });

            const data = await response.json();

            if (data.success) {
                toast({
                    title: "Email Sent! 📧",
                    description: "Please check your inbox for the verification link",
                });
                setStatus('pending');
            } else {
                toast({
                    title: "Failed to Send",
                    description: data.error,
                    variant: "destructive",
                });
            }
        } catch (error) {
            toast({
                title: "Error",
                description: "Could not resend email",
                variant: "destructive",
            });
        } finally {
            setIsResending(false);
        }
    };

    return (
        <>
            <AuthNavbar />

            <div className="min-h-screen flex items-center justify-center p-4 gradient-earth pt-20">
                <Card className="w-full max-w-md shadow-2xl">
                    <CardHeader className="text-center">
                        {status === 'verifying' && (
                            <>
                                <div className="mx-auto mb-4 p-4 rounded-2xl bg-primary/10 w-fit">
                                    <Loader2 className="w-12 h-12 text-primary animate-spin" />
                                </div>
                                <CardTitle className="text-2xl font-bold">Verifying Email...</CardTitle>
                                <CardDescription>Please wait while we verify your email address</CardDescription>
                            </>
                        )}

                        {status === 'success' && (
                            <>
                                <div className="mx-auto mb-4 p-4 rounded-2xl bg-green-100 w-fit">
                                    <CheckCircle2 className="w-12 h-12 text-green-600" />
                                </div>
                                <CardTitle className="text-2xl font-bold text-green-600">Email Verified! 🎉</CardTitle>
                                <CardDescription>{message}</CardDescription>
                            </>
                        )}

                        {status === 'error' && (
                            <>
                                <div className="mx-auto mb-4 p-4 rounded-2xl bg-red-100 w-fit">
                                    <XCircle className="w-12 h-12 text-red-600" />
                                </div>
                                <CardTitle className="text-2xl font-bold text-red-600">Verification Failed</CardTitle>
                                <CardDescription className="text-red-500">{message}</CardDescription>
                            </>
                        )}

                        {status === 'pending' && !token && (
                            <>
                                <div className="mx-auto mb-4 p-4 rounded-2xl bg-primary/10 w-fit">
                                    <Mail className="w-12 h-12 text-primary" />
                                </div>
                                <CardTitle className="text-2xl font-bold">Check Your Email 📧</CardTitle>
                                <CardDescription>We've sent a verification link to your email address</CardDescription>
                            </>
                        )}
                    </CardHeader>

                    <CardContent className="space-y-4">
                        {status === 'success' && (
                            <div className="text-center space-y-4">
                                <p className="text-sm text-muted-foreground">
                                    Redirecting to login page...
                                </p>
                                <Button onClick={() => navigate('/login')} className="w-full">
                                    Go to Login Now
                                </Button>
                            </div>
                        )}

                        {(status === 'error' || (status === 'pending' && !token)) && (
                            <div className="space-y-4">
                                <div>
                                    <label className="text-sm font-medium">Email Address</label>
                                    <input
                                        type="email"
                                        value={email}
                                        onChange={(e) => setEmail(e.target.value)}
                                        placeholder="Enter your email"
                                        className="w-full mt-1 px-3 py-2 border rounded-md"
                                    />
                                </div>

                                <Button
                                    onClick={resendVerification}
                                    disabled={isResending}
                                    className="w-full"
                                    variant="outline"
                                >
                                    {isResending ? (
                                        <>
                                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                            Sending...
                                        </>
                                    ) : (
                                        <>
                                            <RefreshCw className="w-4 h-4 mr-2" />
                                            Resend Verification Email
                                        </>
                                    )}
                                </Button>

                                <p className="text-center text-sm text-muted-foreground">
                                    Already verified?{" "}
                                    <Link to="/login" className="text-primary font-semibold hover:underline">
                                        Sign in
                                    </Link>
                                </p>
                            </div>
                        )}

                        {status === 'pending' && !token && (
                            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 text-sm">
                                <p className="font-medium mb-2">📝 Please note:</p>
                                <ul className="space-y-1 text-muted-foreground">
                                    <li>• Check your spam folder if you don't see the email</li>
                                    <li>• The verification link expires in 24 hours</li>
                                    <li>• Click the link in the email to verify your account</li>
                                </ul>
                            </div>
                        )}
                    </CardContent>
                </Card>
            </div>
        </>
    );
};
